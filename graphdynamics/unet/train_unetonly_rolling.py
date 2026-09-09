"""
Train UNet-only baseline 256x256 — no brain graph.

Usage:
    python3 -u train_unetonly_rolling.py --seed 0 --gpu cuda:0

NOTE the script name: train_unetonly.py (no _rolling) is the SUPERSEDED May
version -- batch 6, mask_ratio 0.26, relative save-dir onto the near-full root
filesystem, no `raw`, no perceptual composite, fp16 encoder. Running it produces
a different experiment. The usage line here used to name it; corrected 2026-08-16.

Always pass --gpu explicitly: it defaults to cuda:0, and the graph arm defaults
to cuda:0 too, so launching both without it co-locates them on one card.
"""

import argparse
import os
import json
import sys
sys.stdout.reconfigure(line_buffering=True)   # added 2026-08-16: v1003 has this;
# without it the per-100-step prints block-buffer at 8 KB, ~40 lines, which at
# ~44 min/epoch is over an hour of apparent silence in a redirected log.
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from model_unetonly import UNetOnly
from data_256 import CelebAHQInpainting


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        for p in vgg.parameters():
            p.requires_grad = False
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target, mask=None):
        with torch.amp.autocast('cuda', enabled=False):
            pred = (pred.float() - self.mean) / self.std
            target = (target.float() - self.mean) / self.std
            loss = 0.0
            x, y = pred, target
            for sl in [self.slice1, self.slice2, self.slice3]:
                x = sl(x)
                y = sl(y)
                if mask is not None:
                    m = F.interpolate(mask.float(), size=x.shape[2:], mode='nearest')
                    inv_m = 1.0 - m
                    n_hidden = inv_m.sum() * x.shape[1]
                    if n_hidden > 0:
                        loss = loss + (inv_m * (x - y)).abs().sum() / n_hidden
                else:
                    loss = loss + F.l1_loss(x, y)
            return loss


def compute_loss(recons, batch, perc_loss_fn=None, raw=None):
    x = batch['x']
    m = batch['m']
    # ---- SIMPLIFIED 2026-08-16 ----
    # Both models return exactly one reconstruction, so T is always 1 and the
    # per-step weight w = (t+1)/T is always 1.0. The loop and the weighting are
    # vestigial, from an earlier model that returned intermediate steps. Kept as
    # a loop so the shape is obvious, with the dead arithmetic removed.
    # was: T = len(recons); ... w = (t + 1) / T; step_loss = w * (...)
    assert len(recons) == 1, f"loss assumes a single reconstruction, got {len(recons)}"
    T = len(recons)

    total = torch.tensor(0.0, device=x.device)
    losses = {}

    for t, recon in enumerate(recons):
        miss = ((1 - m) * (recon - x)).abs().sum() / max((1 - m).sum() * 3, 1)
        # ---- CHANGED 2026-08-16: obs applied to `raw`, matching v1003 ----
        # was: obs = (m * (recon - x)).abs().sum() / max(m.sum() * 3, 1)
        # recon = m*x_obs + (1-m)*raw, so with binary m that reduced to
        # m*(x_obs - x): no dependence on the network, gradient identically zero.
        # v1003 applies it to the raw output; matching so both models optimise
        # the same objective.
        src = raw if raw is not None else recon
        obs = (m * (src - x)).abs().sum() / max(m.sum() * 3, 1)
        total = total + 5.0 * miss + obs

        if t == T - 1:
            losses['miss_final'] = miss.item()
            losses['obs_final'] = obs.item()

    if perc_loss_fn is not None:
        # ---- CHANGED 2026-08-16: composite GROUND TRUTH in the valid region ----
        # was: p_loss = perc_loss_fn(recons[-1].clamp(0, 1), x, mask=m)
        # recon = m*x_obs + (1-m)*raw, so in the observed region recon IS x_obs --
        # which on the training split carries the corruption: a U(0, 0.25)
        # fraction of the observed pixels blanked by the input dropout, plus
        # noise and blur. (Updated 2026-08-16: this used to read "~10% of pixels
        # ... plus noise/brightness/contrast/blur". Both halves went stale the
        # same afternoon -- dropout became a mask-invariant random fraction of
        # the OBSERVED pixels rather than a fixed 10% of the image, and
        # brightness/contrast moved to _photometric, where they act on x and
        # x_obs together and so no longer make x_obs differ from x at all.
        # Whole-channel drop was removed earlier the same day.) The fix below is
        # unaffected by either change: noise, blur and dropout still make the
        # observed region of `recon` differ from `x`. VGG compares that against
        # the clean x, so a network that
        # output the ground truth exactly still scored 1.17 on train (measured)
        # and 0.00 on val. The network cannot reduce it -- those pixels have m=1,
        # so recon does not depend on its output there -- but the gradient is not
        # zero either, because VGG receptive fields at hidden locations overlap
        # the blanked pixels, so it pushes the hidden-region output to compensate
        # for dead neighbours. Compositing x in the valid region is what PConv and
        # standard inpainting losses do: only the hidden region is the network's.
        perc_src = raw if raw is not None else recons[-1]
        perc_in = m * x + (1 - m) * perc_src
        p_loss = perc_loss_fn(perc_in.clamp(0, 1), x, mask=m)
        total = total + 0.1 * p_loss
        losses['perc'] = p_loss.item()

    losses['total'] = total.item()
    return total, losses


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, perc_loss_fn=None,
                    val_loader=None, val_interval=None, best_val_miss=float('inf'), save_dir='checkpoints_unetonly',
                    loader_gen=None):
    model.train()
    running = {'total': 0, 'miss_final': 0, 'obs_final': 0}
    if perc_loss_fn is not None:
        running['perc'] = 0
    n_steps = 0
    t0 = time.time()
    nonfinite_streak = [0]          # list so the guard below can mutate it
    MAX_NONFINITE = 20              # ~0.14% of an epoch's 14,000 steps
    batch_idx = 0                   # counts ALL batches, including skipped ones

    for batch in loader:
        batch_idx += 1
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # ---- CHANGED 2026-08-16: autocast removed, it was dead ----
        # Every block inside both models is wrapped in autocast(enabled=False)
        # -- encoder, decoder, sparse layers, collector, twin path, VGG -- so
        # nothing ever ran in fp16. Verified: all conv outputs are fp32 and the
        # GradScaler recorded zero backoffs in 84,000 steps while its scale grew
        # unbounded toward 1.7e33. Removing the wrapper and the scaler changes
        # no arithmetic; it deletes machinery that only produced a misleading
        # "scaler scale=" figure in the non-finite warning.
        # was: with torch.amp.autocast('cuda', dtype=torch.float16):
        recons, raw = model(batch['x_obs'], batch['m'])
        loss, losses = compute_loss(recons, batch, perc_loss_fn, raw=raw)

        # ---- ADDED 2026-08-16: abort on a non-finite loss ----
        # Without this the run cannot die. Once the forward overflows, every
        # loss is nan, GradScaler skips every step, and its scale decays
        # 256 -> 128 -> ... -> 0; at 0 it can no longer decrease, so even the
        # "scale dropped" signal disappears and the skipped-step counter stops
        # incrementing. Weights freeze while the loop keeps printing nan and
        # writing ~4 GB of checkpoints per epoch for the full 36 h.
        # Observed exactly this at lr 3e-3: nan from step 5, frozen to step 300.
        # A few consecutive nans can be transient (a bad batch under fp16), so
        # tolerate a short run of them and abort only if it persists.
        if not torch.isfinite(loss):
            nonfinite_streak[0] += 1
            print(f"  [WARN] non-finite loss at ep{epoch} batch {batch_idx} "
                  f"(streak {nonfinite_streak[0]}/{MAX_NONFINITE}, "
                  f"grad-clip 1.0)", flush=True)
            if nonfinite_streak[0] >= MAX_NONFINITE:
                raise RuntimeError(
                    f"Aborting: loss non-finite for {MAX_NONFINITE} consecutive "
                    f"steps at epoch {epoch}, batch {batch_idx}. The run is dead; "
                    f"continuing would burn the remaining budget writing nan "
                    f"checkpoints. Most likely cause is too high a learning rate.")
            optimizer.zero_grad(set_to_none=True)
            continue
        nonfinite_streak[0] = 0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in running:
            running[k] += losses.get(k, 0)
        n_steps += 1

        if n_steps % 100 == 0:
            elapsed = time.time() - t0
            avg_miss = running['miss_final'] / n_steps
            perc_str = f" perc={running['perc']/n_steps:.4f}" if 'perc' in running else ""
            print(f"  ep{epoch} step {n_steps}: "
                  f"miss={avg_miss:.4f} "
                  f"total={running['total']/n_steps:.4f}"
                  f"{perc_str} "
                  f"({elapsed:.0f}s)")

        # Rolling checkpoint every 5000 steps
        if n_steps % 5000 == 0:
            # ---- CHANGED 2026-08-16: overwrite one file, matching v1003 ----
            # was: f'rolling_ep{epoch}_step{n_steps}.pt' -- a uniquely-named
            # 1.045 GB file at steps 5000 and 10000 of every epoch, ~157-209 GB
            # over 50 epochs against 317 GB free, and the graph arm needs its
            # share too. v1003 overwrites a single mid.pt; matching that.
            roll_path = os.path.join(save_dir, 'mid.pt')
            torch.save({
                'epoch': epoch,
                'step': n_steps,
                # ---- ADDED 2026-08-16 ----
                # mid.pt used to carry neither 'args' nor 'best_val_miss', so a
                # resume from it silently reset the best to inf (the next epoch
                # then always declared a new best and overwrote the real one) and
                # its scheduler last_epoch was one behind start_epoch, so every
                # later epoch ran at the previous epoch's LR. The marker below
                # makes the resume path handle it correctly. (Until 2026-08-16
                # the marker made it REFUSE outright; the resume path now
                # restarts at ckpt['epoch'] instead of ckpt['epoch']+1, which is
                # correct because scheduler.step() runs at end of epoch, so
                # scheduler.last_epoch == epoch while mid.pt is being written.)
                'mid_epoch': True,
                'best_val_miss': best_val_miss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                # ---- ADDED 2026-08-30 ---- see the note at the epoch-end save.
                'loader_gen_state': (loader_gen.get_state()
                                     if loader_gen is not None else None),
            }, roll_path)
            # ---- SUPERSEDED 2026-08-15 ----
            # Checkpoint pruning disabled 2026-08-15: it deleted all but the last 2 best_*.pt
            # # (and all but 1-3 rolling_*.pt), which is why earlier runs lost their early-epoch
            # # checkpoints. Seed-comparison runs need every epoch retained.
            # import glob as _glob
            # rolls = sorted(_glob.glob(os.path.join(save_dir, 'rolling_*.pt')))
            # while len(rolls) > 3:
            # os.remove(rolls.pop(0))
            # ---- end superseded ----

        # Mid-epoch validation
        if val_loader is not None and val_interval and n_steps % val_interval == 0:
            val_losses = validate(model, val_loader, device, perc_loss_fn)
            print(f"  ** MID-EPOCH VAL ep{epoch} step {n_steps}: "
                  f"val_miss={val_losses['miss_final']:.4f} "
                  f"val_obs={val_losses['obs_final']:.4f}")
            if val_losses['miss_final'] < best_val_miss:
                best_val_miss = val_losses['miss_final']
                ckpt = {
                    'epoch': epoch,
                    'step': n_steps,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                        'val_losses': val_losses,
                    'best_val_miss': best_val_miss,
                }
                torch.save(ckpt, os.path.join(save_dir, f'best_ep{epoch}.pt'))
                print(f"  ** New best val_miss={best_val_miss:.4f} (saved best_ep{epoch}.pt)")
            model.train()

    for k in running:
        running[k] /= max(n_steps, 1)
    return running, best_val_miss


@torch.no_grad()
def validate(model, loader, device, perc_loss_fn=None):
    model.eval()
    running = {'total': 0, 'miss_final': 0, 'obs_final': 0}
    if perc_loss_fn is not None:
        running['perc'] = 0
    n_steps = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # ---- CHANGED 2026-08-16: validation runs in fp32, matching v1003 ----
        # was wrapped in autocast(fp16); train_v1003super.validate() is not, so
        # the two arms' reported val_miss were computed at different precision.
        # Measured effect on a real checkpoint was 0.005%, but there is no reason
        # for the scoring path to differ at all.
        recons, raw = model(batch['x_obs'], batch['m'])
        _, losses = compute_loss(recons, batch, perc_loss_fn, raw=raw)
        for k in running:
            running[k] += losses.get(k, 0)
        n_steps += 1

    for k in running:
        running[k] /= max(n_steps, 1)
    return running


def main():
    parser = argparse.ArgumentParser()
    # ---- CHANGED 2026-09-09: no hardcoded dataset path ----
    # was: default='/home/vahid/data/celebahq256' -- an absolute path on the
    # author's machine, so the documented command only ran for one person.
    # None now means: use $TENSORMIND_CELEBAHQ, then a cached copy, then
    # download a pinned one. See celebahq.py.
    parser.add_argument('--data', type=str, default=None,
                        help="CelebA-HQ 256 root, holding train/ and validation/. "
                             "Omit to use $TENSORMIND_CELEBAHQ, a cached copy, or "
                             "to download a pinned one.")
    parser.add_argument('--no-download', action='store_true',
                        help="Never download the dataset; fail instead.")
    parser.add_argument('--batch', type=int, default=2)  # was 6; matched across models (2026-08-15)
    # ---- REVERTED 2026-08-16, same day: back to 3e-4. The 3e-3 change was WRONG.
    # It rested on comparing the "lr=" figures in the two training logs, but both
    # print scheduler.get_last_lr()[0], i.e. param_groups[0], and the two scripts
    # order their groups differently:
    #   graph model: group0 = BRAIN params at args.lr * brain_lr_mult(10) -> 3e-3
    #   this script : group0 = encoder/other params at args.lr * 1        -> 3e-4
    # Both models therefore trained their encoders at 3e-4. (Those encoders are
    # identical apart from `mid`, which only this baseline has -- corrected
    # 2026-08-16, an earlier version of this comment said byte-identical.)
    # The base LR was already matched; there was never a 10x discrepancy.
    # Setting 3e-3 here would have put this encoder at 10x the graph model's.
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--mask-ratio', type=float, default=0.25)  # match graph models (was 0.26)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)  # recorded in ckpt['args'] (added 2026-08-15)
    parser.add_argument('--gpu', type=str, default='cuda:0')  # match other scripts (added 2026-08-15)
    parser.add_argument('--val-subset', type=str, default='selection',
                        choices=['selection', 'test', 'all'],
                        help="which half of the 2,000 validation images to validate on. "
                             "'selection' (default) keeps the reporting half unseen; "
                             "eval_benchmark_v2.py reports on 'test'.")
    parser.add_argument('--val-split', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'val_test_split_v1.json'))
    parser.add_argument('--resume', type=str, default=None)
    # ---- CHANGED 2026-08-16: default save-dir moved to the T7 ----
    # 'checkpoints_unetonly' is relative, so it resolved to
    # /home/vahid/experimentbrain/checkpoints_unetonly on / -- which is 99% full
    # (11 GB free of 916 GB). One checkpoint for this model is 1,045,535,738
    # bytes, and with pruning disabled each epoch writes rolling_step5000,
    # rolling_step10000, rolling_end and (while improving) best_ep{n}, about
    # 4.2 GB/epoch. The run would have died of ENOSPC during epoch 2.
    # That directory also already holds best_ep36.pt from an April run, so
    # writing into it would have mixed two runs together.
    # was: default='checkpoints_unetonly')
    parser.add_argument('--save-dir', type=str,
                        # ---- CHANGED 2026-08-16: new dir for the restarted run ----
                        # Never reuse a save-dir. ckpt_unet_s0 holds the run
                        # started 15:33 under the pre-photometric-fix pipeline.
                        # was: default='/media/vahid/T7 Shield/ckpt_unet_s0')
                        default='/media/vahid/T7 Shield/ckpt_unet_s0_r4')
    args = parser.parse_args()

    import random as _random, numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Seed: {args.seed}  (torch.initial_seed={torch.initial_seed()})", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- ADDED 2026-09-09 ---- resolve the dataset, downloading if needed.
    from celebahq import resolve as _resolve_celebahq
    args.data = _resolve_celebahq(args.data, download=not args.no_download)
    print(f"CelebA-HQ root: {args.data}")

    train_ds = CelebAHQInpainting(args.data, split='train', mask_ratio=args.mask_ratio,
                                   irregular=True)
    val_ds = CelebAHQInpainting(args.data, split='validation', mask_ratio=args.mask_ratio,
                                 irregular=True)
    # ---- ADDED 2026-08-16: validate on the SELECTION half only ----
    # CelebA-HQ is 28,000 train + 2,000 validation with no unused pool, so the
    # same 2,000 images were both choosing best_ep*.pt (once per epoch, 50 times)
    # and being reported -- selection on the test set. val_test_split_v1.json
    # partitions them 1000/1000 with seed 20260815. Training now sees only the
    # selection half; eval_benchmark_v2.py defaults to --val-subset test, so the
    # reported images are never used to pick a checkpoint.
    if args.val_subset != 'all':
        import json as _json
        with open(args.val_split) as _f:
            _sp = _json.load(_f)
        _idx = _sp['selection_idx'] if args.val_subset == 'selection' else _sp['test_idx']
        val_ds = torch.utils.data.Subset(val_ds, _idx)
        print(f"Validation restricted to the '{args.val_subset}' half: {len(_idx)} images "
              f"(split {args.val_split}, seed {_sp['rng_seed']})")
    _g = torch.Generator(); _g.manual_seed(args.seed)
    def _winit(wid):
        # ---- FIXED 2026-08-15 ---- issue 9: this seeded Python `random` and GLOBAL numpy, neither of
        # which data_256.py uses - it draws masks from np.random.default_rng(...)
        # and torch.randint. Mask seeding therefore worked only by accident, via
        # the DataLoader `generator=`. Seed torch in the worker as well, and give
        # each epoch a different stream so masks are not identical every epoch.
        import random as _r, numpy as _n, torch as _t
        _s = args.seed * 1000 + wid
        _r.seed(_s); _n.random.seed(_s)
        # ---- CHANGED 2026-08-16 ----
        # was: _t.manual_seed(_s)
        # PyTorch seeds each worker with base_seed + worker_id, where base_seed is
        # redrawn from the DataLoader generator EVERY epoch, and then calls this
        # function -- so a constant here overwrote the only per-epoch entropy in
        # the torch stream. data_256.py draws its mask offset as
        # `idx + torch.randint(0, 100000, (1,))`, so the offset table was frozen
        # to the same 14,000 values for all 50 epochs (masks still varied, but
        # only because shuffling moved idx between slots). RandomHorizontalFlip
        # was likewise slot-determined. Deriving from torch.initial_seed() keeps
        # the run reproducible while restoring per-epoch variation.
        _t.manual_seed(_t.initial_seed() + wid)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              generator=_g, worker_init_fn=_winit,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = UNetOnly().to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # ---- CHANGED 2026-08-16: mid-epoch resume RESTORED ----
        # This used to raise SystemExit. The stated reason -- "its scheduler is
        # one epoch behind the epoch it would restart at" -- was true only
        # because the line below unconditionally did `ckpt['epoch'] + 1`.
        #
        # mid.pt stores 'epoch': epoch while the epoch is still in progress, and
        # scheduler.step() runs at the END of an epoch (line ~477 / ~638), so
        # scheduler.last_epoch == epoch at the moment mid.pt is written.
        # Restarting AT ckpt['epoch'] with the saved scheduler is therefore
        # exactly right; the only cost is redoing the part of that epoch already
        # done. The capability was deleted while the ~1 GB write, twice per
        # epoch, was kept.
        #
        # This matters on this box: two runs died today (one on a bad edit of
        # mine, one killed deliberately), the hardware has a recorded Xid 79
        # history, and these are 35 h and 46 h runs. Without this, a crash at
        # hour 40 loses everything; with it, it loses part of one epoch.
        # was: if <ckpt>.get('mid_epoch'):
        # was:     raise SystemExit("Refusing to resume from a mid-epoch checkpoint ...")
        _mid = bool(ckpt.get('mid_epoch'))
        model.load_state_dict(ckpt['model'])
        # restart AT the interrupted epoch, not after it
        start_epoch = ckpt['epoch'] if _mid else ckpt['epoch'] + 1
        if _mid:
            print(f"  Mid-epoch checkpoint (was at step {ckpt.get('step', '?')} "
                  f"of epoch {ckpt['epoch']}); restarting that epoch from its "
                  f"beginning with the saved optimizer and scheduler.")
        best_val_miss = ckpt.get('best_val_miss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_miss = float('inf')

    # ---- CHANGED 2026-08-16: decoder multiplier 0.3 -> 1.0 ----
    # Matches the graph arm. The 0.3x came from the v6/v7 warm-start lineage and
    # has no justification in a from-scratch run; it was holding 46,829,699
    # params (62% of this model) at 9e-5. Single rate for the whole network now.
    # Two param groups kept so the split stays visible in the log.
    # Reported for the log only -- the optimiser groups below are decay/no-decay,
    # not decoder/other, since every module now trains at the same rate. Kept so
    # the split stays visible and comparable with the graph arm's breakdown.
    n_dec = sum(p.numel() for n, p in model.named_parameters() if n.startswith('decoder'))
    n_oth = sum(p.numel() for n, p in model.named_parameters() if not n.startswith('decoder'))
    print(f"Decoder params: {n_dec:,} (1x LR)")
    print(f"Other params: {n_oth:,} (1x LR)")

    # ---- CHANGED 2026-08-16: exclude norms and biases from weight decay ----
    # was: AdamW([...], weight_decay=1e-4) -- uniform wd over every parameter,
    # including all GroupNorm affine scales and every bias. Decaying a
    # normalisation scale pulls it toward zero for no reason; standard practice
    # excludes norms and biases. Matches the change made to the graph arm, which
    # additionally had to rescale wd per group because of its 10x LR group.
    BASE_WD = 1e-4

    def _no_decay(name):
        return name.endswith('.bias') or 'norm' in name.lower()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if _no_decay(n) else decay).append(p)
    print(f"  weight decay: {sum(p.numel() for p in decay):,} params @ {BASE_WD:.1e}  |  "
          f"no-decay (norms+biases): {sum(p.numel() for p in no_decay):,}")

    optimizer = torch.optim.AdamW([
        {'params': decay,    'lr': args.lr, 'weight_decay': BASE_WD},
        {'params': no_decay, 'lr': args.lr, 'weight_decay': 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # GradScaler removed 2026-08-16 -- see the note in train_one_epoch. Nothing
    # in either model runs in fp16, so it never scaled or skipped anything.

    if args.resume:
        # ---- CHANGED 2026-08-16: reuse the checkpoint already loaded above ----
        # was: ckpt_data = torch.load(args.resume, ...) -- a second full load of
        # the same ~1 GB file, and ckpt_data stayed referenced by main() for the
        # whole run, pinning 1 GB of GPU memory that was never used again.
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        # ---- ADDED 2026-08-30 ---- restore the data stream, not just the model.
        # _g was re-seeded with args.seed above; without this the resumed run
        # replays epoch 0's shuffle order and masks. set_state mutates in place,
        # so train_loader's existing reference to _g picks this up.
        _gs = ckpt.get('loader_gen_state')
        if _gs is not None:
            _g.set_state(_gs)
            print("  DataLoader generator state restored — data stream continues.")
        else:
            print("  [WARN] checkpoint predates 2026-08-30 and carries no "
                  "DataLoader generator state; the data stream will restart from "
                  "epoch 0's shuffle order and masks. Epochs will be repeated.")

    perc_loss_fn = PerceptualLoss().to(device).eval()
    print("Perceptual loss enabled (VGG16, weight=0.1)")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_losses, best_val_miss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, perc_loss_fn,
            # val_interval=None disables mid-epoch validation (was 5000).
            # Disabled 2026-08-16: at 14,000 steps/epoch it fired twice per epoch,
            # cost ~2h per run, and its readings were systematically optimistic
            # (measured +0.0131 mean drift vs end-of-epoch on the baseline) while
            # being exactly what set best_val_miss. The reported statistic is a
            # last-K-epoch mean of end-of-epoch values, so they were pure cost.
            val_loader=val_loader, val_interval=None, best_val_miss=best_val_miss,
            save_dir=args.save_dir, loader_gen=_g)
        val_losses = validate(model, val_loader, device, perc_loss_fn)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch}: "
              f"train_miss={train_losses['miss_final']:.4f} "
              f"val_miss={val_losses['miss_final']:.4f} "
              f"val_obs={val_losses['obs_final']:.4f} "
              f"lr[encoder]={scheduler.get_last_lr()[0]:.2e} "
              f"({elapsed:.0f}s)")

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_miss': best_val_miss,
            'args': vars(args),
            # ---- ADDED 2026-08-30 ---- resume replayed the data stream.
            # model/optimizer/scheduler were saved but the DataLoader generator
            # was not, and main() re-seeds it with args.seed at process start.
            # PyTorch draws each epoch's worker base seed from that generator,
            # and data_256.py derives every mask, flip and shuffle order from
            # the worker torch stream -- so a resume rewound the whole data
            # pipeline to epoch 0. The claim in the resume block that "the only
            # cost is redoing the part of that epoch already done" was wrong.
            # Found 2026-08-30 on feeder1004 seed 2, which was resumed at ep16
            # after a hardware crash and replayed the shuffle orders and masks
            # of its own ep0-15 in ep16-31: 34 distinct data streams over 50
            # epochs instead of 50. No UNet run was ever resumed, so no UNet
            # result is affected -- this is a latent bug being closed, not a
            # correction to anything already measured.
            # Verified: with this, a resumed stream is bit-identical to an
            # uninterrupted one; without it, the first epoch after a restart
            # reproduces epoch 0 exactly.
            'loader_gen_state': _g.get_state(),
        }
        # ---- ACTUALLY FIXED 2026-08-16 ---- issue 10.
        # The 2026-08-15 comment claimed best_val_miss was "updated first"; the
        # code still saved latest.pt and rolling_ep*.pt BEFORE the comparison
        # below, so every one of those files recorded the previous epoch's value
        # and a resume from latest.pt would immediately re-declare a spurious
        # new best. The update now genuinely happens first.
        is_best = val_losses['miss_final'] < best_val_miss
        if is_best:
            best_val_miss = val_losses['miss_final']
        ckpt['best_val_miss'] = best_val_miss

        torch.save(ckpt, os.path.join(args.save_dir, 'latest.pt'))
        # per-epoch history, matching train_v1001super.py (added 2026-08-15)
        torch.save(ckpt, os.path.join(args.save_dir, f'rolling_ep{epoch}_end.pt'))

        if is_best:
            # ---- CHANGED 2026-08-16: pointer, not a second copy ----
            # best_ep{n}.pt was a byte-identical duplicate of the
            # rolling_ep{n}_end.pt written three lines above: same `ckpt` dict,
            # same epoch, saved back to back. Over 50 epochs that is up to 50
            # extra full checkpoints per arm. At 1,045,535,738 bytes per checkpoint that is
            # about 52 GB of pure duplication per arm, against 315 GB free on the
            # T7 with two arms running and two more seeds to come.
            # Hardlinks and symlinks are both ENOSYS on that volume (fuseblk),
            # so the duplicate is replaced by a small JSON pointer.
            # Mid-epoch best saving (line ~246) is unaffected: it runs only
            # when val_interval is set, and it is None for these runs.
            # was: torch.save(ckpt, os.path.join(args.save_dir, f'best_ep{epoch}.pt'))
            best_ptr = os.path.join(args.save_dir, 'best.json')
            with open(best_ptr, 'w') as _f:
                json.dump({'epoch': epoch,
                           'val_miss': float(best_val_miss),
                           'checkpoint': f'rolling_ep{epoch}_end.pt'}, _f, indent=1)
            print(f"  ** New best val_miss={best_val_miss:.4f} "
                  f"(best.json -> rolling_ep{epoch}_end.pt)")


if __name__ == '__main__':
    main()
