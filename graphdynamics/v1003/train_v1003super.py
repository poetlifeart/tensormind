"""
Train V1003Super — v1001super with the bottleneck removed, the graph fed by
down4, projected t4/t5 readouts, and obs-on-raw.

Usage:
    python -u train_v1003super.py --seed 0 --gpu cuda:1 --from-scratch

Always pass --gpu explicitly: it defaults to cuda:0 and so does the UNet
baseline script, so launching both without it co-locates them on one card.
"""

import argparse
import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from model_v1003super import RecurrentBrainNetV7
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
        # ---- CHANGED 2026-08-15 (v1003 #8) ----
        # was: obs = (m * (recon - x)).abs().sum() / max(m.sum() * 3, 1)
        # recon = m*x_obs + (1-m)*raw, so with binary m the observed term reduced to
        # m*(x_obs - x): no dependence on the network output, gradient identically
        # zero, and val_obs printed 0.0000 in every log. Applied to `raw` instead,
        # which is the valid-region term used by PConv and standard inpainting losses.
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


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch,
                    perc_loss_fn=None, val_loader=None, val_interval=None,
                    best_val_miss=float('inf'), save_dir='checkpoints_v7',
                    loader_gen=None,
                    ):  # alpha_start/alpha_end/total_steps removed 2026-08-16, see below
    model.train()
    running = {'total': 0, 'miss_final': 0, 'obs_final': 0}
    if perc_loss_fn is not None:
        running['perc'] = 0
    n_steps = 0
    t0 = time.time()
    # ---- ADDED 2026-08-16: abort on a persistently non-finite loss ----
    # Matches the guard in train_unetonly_rolling.py so the two arms fail the
    # same way. Without it a NaN run cannot die: GradScaler skips every step,
    # its scale decays to 0 and can no longer decrease, so even the skipped
    # counter stops moving while the loop keeps writing checkpoints for 47 h.
    # Isolated overflows are survivable, so only a sustained run aborts.
    nonfinite_streak = [0]
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
                    f"checkpoints.")
            optimizer.zero_grad(set_to_none=True)
            continue
        nonfinite_streak[0] = 0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in running:
            running[k] += losses.get(k, 0)
        n_steps += 1

        # ---- REMOVED 2026-08-16 ----
        # This ramp wrote model.collector_grad_scale, which nothing in
        # model_v1003super.py reads, and main() never passed alpha_start /
        # alpha_end so it never ran. Dead in two independent ways.
        # was: if alpha_start is not None and alpha_end is not None:
        # was:     frac = min(n_steps / total_steps, 1.0)
        # was:     model.collector_grad_scale = alpha_start + (alpha_end - alpha_start) * frac

        if n_steps % 100 == 0:
            elapsed = time.time() - t0
            avg_miss = running['miss_final'] / n_steps
            perc_str = f" perc={running['perc']/n_steps:.4f}" if 'perc' in running else ""
            print(f"  ep{epoch} step {n_steps}: "
                  f"miss={avg_miss:.4f} "
                  f"total={running['total']/n_steps:.4f}"
                  f"{perc_str} "
                  f"({elapsed:.0f}s)")

        if n_steps % 5000 == 0:
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
            }, os.path.join(save_dir, 'mid.pt'))

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
    # ---- REVERTED 2026-08-16: back to the parent-clustered supernodes ----
    # Fix #14 re-clustered the FEEDER graph, on the reasoning that the model runs
    # on the feeder. That was wrong: the feeder's extra 6,500 edges were CREATED
    # from the parent's partition (invert146_feeder.py realises each refined
    # quotient edge at parent level), so the parent partition is the one those
    # edges belong to. Verified by re-running the identical coarsen():
    #   ARI(v1001 supernode ids, fresh coarsen of parent) = 1.0000  (VI = 0)
    #   ARI(v1003 supernode ids, fresh coarsen of parent) = 0.6845  (VI = 2.56)
    # Clustering the feeder gives attention groups unrelated to the partition
    # that generated the feeder's own edges. Reverting so v1003 tests only its
    # four intended changes (mid removed + down4->graph, new twin path,
    # attn_1 on iteration 0, observed-loss on raw).
    # default='/home/vahid/experimentbrain/graph_v1003_feeder_supernode.npz')
    parser.add_argument('--graph', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'graph_brain_mild_v146_feeder_supernode.npz'))
    parser.add_argument('--data', type=str,
                        default='/home/vahid/data/celebahq256')
    parser.add_argument('--v6-ckpt', type=str, default='')
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--batch', type=int, default=2)  # was 4; matched across models (2026-08-15)
    # ---- REVERTED 2026-08-16, same day: back to 3e-4. The 3e-3 change was WRONG.
    # It came from reading "lr=3.00e-03" in the v1001super logs as the base LR.
    # That line prints scheduler.get_last_lr()[0] = param_groups[0], which in this
    # script is the BRAIN group at args.lr * brain_lr_mult (10x). Base LR was 3e-4
    # in every graph run, confirmed by the recorded args in every checkpoint.
    # With --lr 3e-3 the brain group would have run at 3e-2, 100x its trained value.
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--brain-lr-mult', type=float, default=10.0)
    # ---- CHANGED 2026-08-16: decoder multiplier 0.3 -> 1.0 ----
    # The 0.3x is a warm-start hyperparameter: it dates from the v6/v7 lineage
    # where the decoder was initialised from a checkpoint and needed protecting.
    # These runs are --from-scratch, so it was throttling 46,829,699 params --
    # 62% of the model -- to 9e-5, falling to 8.6e-6 by epoch 40, for no reason.
    # The graph keeps its 10x (it is tiny, sparse, and that multiplier was
    # derived for it); everything else is now even at the base rate.
    # was: parser.add_argument('--decoder-lr-mult', type=float, default=0.3)
    parser.add_argument('--decoder-lr-mult', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n-iters', type=int, default=5)
    parser.add_argument('--mask-ratio', type=float, default=0.25)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)  # recorded in ckpt['args'] (added 2026-08-15)
    parser.add_argument('--val-subset', type=str, default='selection',
                        choices=['selection', 'test', 'all'],
                        help="which half of the 2,000 validation images to validate on. "
                             "'selection' (default) keeps the reporting half unseen; "
                             "eval_benchmark_v2.py reports on 'test'.")
    parser.add_argument('--val-split', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'val_test_split_v1.json'))
    parser.add_argument('--resume', type=str, default=None)
    # ---- CHANGED 2026-08-16: own save-dir ----
    # Default pointed at the v1001 checkpoint directory name. Never reuse a
    # save-dir across runs -- latest.pt would be overwritten.
    # default='/home/vahid/experimentbrain/checkpoints_v1001super')
    parser.add_argument('--save-dir', type=str,
                        # ---- CHANGED 2026-08-16: new dir for the restarted run ----
                        # Never reuse a save-dir. ckpt_v1003_s0 holds the run
                        # started 15:35 under the pre-photometric-fix pipeline
                        # and with twin_path at 1x.
                        # was: default='/media/vahid/T7 Shield/ckpt_v1003_s0')
                        default='/media/vahid/T7 Shield/ckpt_v1003_s0_r4')
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
        # ---- FIXED 2026-08-15 ---- issue 9: this seeded Python `random` and
        # GLOBAL numpy, neither of which data_256.py uses - it draws masks from
        # np.random.default_rng(...) and torch.randint. Mask seeding worked only
        # by accident via the DataLoader `generator=`. Seed torch here too.
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

    model = RecurrentBrainNetV7(args.graph, brain_channels=32, n_iters=args.n_iters).to(device)

    ckpt_data = None
    if args.resume:
        ckpt_data = torch.load(args.resume, map_location=device, weights_only=False)
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
        _mid = bool(ckpt_data.get('mid_epoch'))
        missing, unexpected = model.load_state_dict(ckpt_data['model'], strict=False)
        if missing:
            print(f"  New params (randomly init): {missing}")
        # restart AT the interrupted epoch, not after it
        start_epoch = ckpt_data['epoch'] if _mid else ckpt_data['epoch'] + 1
        if _mid:
            print(f"  Mid-epoch checkpoint (was at step {ckpt_data.get('step', '?')} "
                  f"of epoch {ckpt_data['epoch']}); restarting that epoch from its "
                  f"beginning with the saved optimizer and scheduler.")
        best_val_miss = ckpt_data.get('best_val_miss', float('inf'))
        saved_args = ckpt_data.get('args', {})
        # ---- FIXED 2026-08-15 ---- issue 7: mid-epoch best_ep*.pt are written WITHOUT 'args', so
        # saved_args was empty and .get('from_scratch', False) silently returned
        # False -> the encoder got frozen on resume, killing 40M trainable params
        # without any message. Default to True and say so when args are absent.
        # was: if saved_args.get('from_scratch', False):
        if 'args' not in ckpt_data:
            print("  [WARN] checkpoint has no 'args' (mid-epoch save); "
                  "assuming from_scratch=True so the encoder is not frozen")
        if saved_args.get('from_scratch', True):
            args.from_scratch = True
        print(f"Resumed from epoch {start_epoch} (from_scratch={args.from_scratch})")
    elif args.from_scratch:
        print("Training from scratch — no pretrained weights.")
        start_epoch = 0
        best_val_miss = float('inf')
    else:
        # ---- ADDED 2026-08-16: fail clearly instead of torch.load('') ----
        # --from-scratch is store_true (default False), --v6-ckpt defaults to ''
        # and --resume to None, so launching with none of the three fell through
        # to torch.load('') and died with a bare FileNotFoundError after the
        # model and dataset had already been built. v1003 is only ever trained
        # from scratch (standing instruction: always from-scratch, no resumes),
        # so say so plainly.
        if not args.v6_ckpt:
            raise SystemExit(
                "No initialisation specified. Pass --from-scratch (the intended "
                "mode for v1003), or --resume <ckpt> to continue a run, or "
                "--v6-ckpt <ckpt> to warm-start from a V6 checkpoint.")
        ckpt = torch.load(args.v6_ckpt, map_location=device, weights_only=False)
        v6_state = ckpt['model'] if 'model' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(v6_state, strict=False)
        print(f"Loaded V6 checkpoint: {args.v6_ckpt}")
        print(f"  Missing keys (new in V7): {len(missing)} — {missing}")
        print(f"  Unexpected keys: {len(unexpected)} — {unexpected}")
        start_epoch = 0
        best_val_miss = float('inf')

    if not args.from_scratch:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen.")

    brain_names = {'collector', 'layer_12', 'layer_13', 'layer_23', 'layer_32',
                   'layer_31', 'ln1', 'ln2', 'ln3', 'ln_fb', 'ln_fb1', 'twin_path',
                   'attn_1', 'attn_2', 'attn_3'}
    brain_params = []
    decoder_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split('.')[0]
        # ---- ADDED 2026-08-16: the t4/t5 projections go in the 1x group ----
        # twin_path is in brain_names, so when it was ~49-65k params of 1x1 heads
        # the 10x brain multiplier was harmless. proj_t4 (1,844,224) and proj_t5
        # (461,056) made it 2,370,752 -- larger than the sparse graph itself
        # (596,260 weights across the five sparse layers) -- and they would have trained at 3e-4*10 = 3e-3 while the
        # UNet baseline's functional counterpart (encoder.mid, the module the
        # graph replaces) trains at 1x = 3e-4. Matching them, so the readout
        # into the decoder learns at the same rate on both sides. The 10x stays
        # on the sparse graph, the LayerNorms, the attention, the collector and
        # the 1x1 heads, exactly as in v1001.
        # ---- WIDENED 2026-08-16: the WHOLE twin path goes in the 1x group ----
        # First pass moved only proj_t4/proj_t5, which left head3/head4/head5
        # (16,896 each) at 10x -- and those are the last layer before the
        # decoder's three deepest inputs, i.e. exactly where the substituted
        # module meets the decoder. The baseline's counterpart, encoder.mid,
        # trains at 1x. The whole graph->decoder readout now trains at 1x too;
        # the 10x multiplier stays on the sparse graph itself (the 5 sparse
        # layers, the LayerNorms, the supernode attention and the collector),
        # which is what it was introduced for.
        # ---- REVERSED 2026-08-16 (c): the whole twin path goes back to 10x ----
        # The two notes above are kept as the record of how this was reasoned.
        # The reasoning was applied to only one of the graph's two interfaces.
        # `collector` -- 63,680 params of Conv2d(ch, 32, 1) that project the
        # encoder's feature maps INTO the graph -- is the structural mirror of
        # twin_path's heads, which project OUT of it, and collector was left in
        # the 10x group at line ~474 (`top in brain_names`). So the graph's input
        # projections were training ten times faster than its output projections,
        # with no argument for the asymmetry; the encoder.mid analogy applies
        # equally to both or to neither.
        #
        # Resolved by treating both as graph: collector and twin_path at 10x.
        #
        # ---- REVERTED 2026-08-16 (d): twin_path goes BACK to 1x. MEASURED. ----
        # The reversal above was argued from symmetry and it is wrong. It ran for
        # 3.5 h and the graph arm went from beating the baseline to trailing it
        # by ~20% on BOTH train_miss and val_miss -- a training-objective
        # regression, so not a generalisation effect.
        #
        #   matched step 2000 of epoch 0, same data pipeline:
        #     unet   0.1511 -> 0.1470   (-2.7%, the data fix)
        #     v1003  0.1353 -> 0.1780  (+31.6%, the data fix AND this change)
        #
        #   epoch-level val_miss, v1003 / unet:
        #     twin 1x   ep0 0.1229/0.1316  ep1 0.1097/0.1132  ep2 0.0986/0.1202
        #               (graph ahead by 6.6% / 3.1% / 18.0%)
        #     twin 10x  ep0 0.1081/0.0871  ep1 0.0884/0.0741  ep2 0.0818/0.0685
        #               (graph behind by 24% / 19% / 19%)
        #
        # Mechanism: head0/head1/head2 are zero-initialised (model_v1003super.py
        # :534-536) and are added straight into the decoder's high-resolution
        # skips, `skips_mod = (s0 + t0, s1 + t1, s2 + t2)` (:769). At 3e-3 they
        # leave zero ten times faster and inject unnormalised noise at 256^2,
        # 128^2 and 64^2. val_obs -- the quantity those three heads perturb --
        # is what reversed hardest: 0.0498/0.0503/0.0524 (better than baseline)
        # became 0.1007/0.0758/0.0669 (49-62% worse).
        #
        # Why the symmetry argument fails: the collector's output is consumed as
        # `a1_base = self.ln1(self.collector(...))` (model_v1003super.py:733), so
        # a LayerNorm removes its scale immediately and a 10x LR on it is cheap.
        # Nothing normalises twin_path's output -- it lands directly in decoder
        # feature maps, where magnitude matters. The two interfaces mirror each
        # other structurally and not numerically. collector stays at 10x.
        if name.startswith('twin_path'):
            encoder_params.append(param)
        elif top == 'encoder':
            encoder_params.append(param)
        elif top in brain_names:
            brain_params.append(param)
        else:
            decoder_params.append(param)
        # ---- the 10x arrangement this replaces, kept as the record ----
        # Argued from "the graph's gradients are weak (measured RMS
        # 6.6e-7..2.5e-6 against the encoder's 8.3e-5) and the readout carries
        # those same weak gradients". The measurement is real but it was taken
        # on the SPARSE GRAPH, not on proj_t4/head*, which sit one op from the
        # decoder and carry decoder-scale gradients. Falsified by the run above.
        # was: if top == 'encoder':
        # was:     encoder_params.append(param)
        # was: elif top in brain_names:
        # was:     brain_params.append(param)
        # was: else:
        # was:     decoder_params.append(param)

    n_brain = sum(p.numel() for p in brain_params)
    n_decoder = sum(p.numel() for p in decoder_params)
    n_encoder = sum(p.numel() for p in encoder_params)
    print(f"Brain params: {n_brain:,} ({args.brain_lr_mult}x LR)")
    print(f"Decoder params: {n_decoder:,} ({args.decoder_lr_mult:g}x LR)")
    if encoder_params:
        # ---- CHANGED 2026-08-16: name the group honestly ----
        # The twin path moved back to the 10x brain group (see note above), so
        # this group is the encoder alone again. The brain line is the one that
        # now needs the breakdown: it holds the sparse graph plus BOTH graph
        # interfaces, and printing one number hid where the readout sits.
        n_twin = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith('twin_path') and p.requires_grad)
        n_coll = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith('collector') and p.requires_grad)
        print(f"Encoder+twin params: {n_encoder:,} (1x LR)  "
              f"[encoder {n_encoder - n_twin:,} + twin_path {n_twin:,}]")
        print(f"  [brain {n_brain:,} = graph {n_brain - n_coll:,} + "
              f"collector {n_coll:,} (in, LayerNorm'd) @ {args.brain_lr_mult}x; "
              f"twin_path {n_twin:,} (out, unnormalised) @ 1x]")

    # ---- CHANGED 2026-08-16: weight decay done properly ----
    # Two inherited defects, both fixed here:
    #  (a) weight_decay=1e-4 was applied to EVERY parameter, including all 187
    #      GroupNorm/LayerNorm scales and biases (53,667 params). Decaying a
    #      normalisation scale pulls it toward zero for no reason; standard
    #      practice excludes norms and biases.
    #  (b) AdamW's decoupled decay is lr*wd, so the 10x-LR brain group was also
    #      being decayed 10x harder than everything else -- 3e-7/step against
    #      3e-8. Nobody chose that; it is a side effect of the LR multiplier,
    #      and it works directly against the 10x that exists because the graph's
    #      gradients are weak (measured RMS 6.6e-7..2.5e-6 vs the encoder's
    #      8.3e-5). Per-group wd is now scaled by 1/mult so the effective decay
    #      per step is identical across groups.
    # was: AdamW(param_groups, weight_decay=1e-4)   # uniform wd, all params
    BASE_WD = 1e-4

    def _no_decay(name):
        # norms and biases: every LayerNorm/GroupNorm affine, every .bias
        return name.endswith('.bias') or 'norm' in name.lower() or '.ln' in name or name.startswith('ln')

    nd = {n for n, p in model.named_parameters() if p.requires_grad and _no_decay(n)}
    split = lambda plist, want_nd: [p for n, p in model.named_parameters()
                                    if p.requires_grad and any(p is q for q in plist)
                                    and ((n in nd) == want_nd)]
    param_groups = []
    for tag, plist, mult in [('brain', brain_params, args.brain_lr_mult),
                             ('decoder', decoder_params, args.decoder_lr_mult),
                             ('encoder', encoder_params, 1.0)]:
        if not plist:
            continue
        lr = args.lr * mult
        # scale wd by 1/mult so lr*wd -- the actual decay applied per step -- is
        # the same for every group
        wd = BASE_WD / mult
        d = split(plist, False)
        n_ = split(plist, True)
        if d:
            param_groups.append({'params': d, 'lr': lr, 'weight_decay': wd})
        if n_:
            param_groups.append({'params': n_, 'lr': lr, 'weight_decay': 0.0})
        print(f"  {tag:13s} lr {lr:.2e}  decay {sum(x.numel() for x in d):,} @ wd {wd:.1e}"
              f"  |  no-decay {sum(x.numel() for x in n_):,}")

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # GradScaler removed 2026-08-16 -- see the note in train_one_epoch. Nothing
    # in either model runs in fp16, so it never scaled or skipped anything.

    if ckpt_data is not None:
        optimizer.load_state_dict(ckpt_data['optimizer'])
        scheduler.load_state_dict(ckpt_data['scheduler'])
        # ---- ADDED 2026-08-30 ---- restore the data stream, not just the model.
        # _g was re-seeded with args.seed above; without this the resumed run
        # replays epoch 0's shuffle order and masks. set_state mutates in place,
        # so train_loader's existing reference to _g picks this up.
        _gs = ckpt_data.get('loader_gen_state')
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
        # ---- REMOVED 2026-08-16: printed every epoch, described a mechanism
        # (a collector gradient ramp) that does not exist in this model.
        # was: print(f"  full gradient (alpha=1.0)")
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
              f"lr[brain 10x]={scheduler.get_last_lr()[0]:.2e} "
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
            # epochs instead of 50. No v1003 run was ever resumed, so no v1003
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
        # below, so every one of those files recorded the value from the previous
        # epoch and a resume from latest.pt would immediately re-declare a
        # spurious new best. The update now genuinely happens first.
        is_best = val_losses['miss_final'] < best_val_miss
        if is_best:
            best_val_miss = val_losses['miss_final']
        ckpt['best_val_miss'] = best_val_miss

        latest_path = os.path.join(args.save_dir, 'latest.pt')
        torch.save(ckpt, latest_path)
        roll_path = os.path.join(args.save_dir, f'rolling_ep{epoch}_end.pt')
        torch.save(ckpt, roll_path)

        if is_best:
            # ---- CHANGED 2026-08-16: pointer, not a second copy ----
            # best_ep{n}.pt was a byte-identical duplicate of the
            # rolling_ep{n}_end.pt written three lines above: same `ckpt` dict,
            # same epoch, saved back to back. Over 50 epochs that is up to 50
            # extra full checkpoints per arm. At ~0.9 GB per checkpoint that is
            # about 45 GB of pure duplication per arm, against 315 GB free on the
            # T7 with two arms running and two more seeds to come.
            # Hardlinks and symlinks are both ENOSYS on that volume (fuseblk),
            # so the duplicate is replaced by a small JSON pointer.
            # Mid-epoch best saving (line ~232) is unaffected: it runs only
            # when val_interval is set, and it is None for these runs.
            # was: torch.save(ckpt, os.path.join(args.save_dir, f'best_ep{epoch}.pt'))
            best_ptr = os.path.join(args.save_dir, 'best.json')
            with open(best_ptr, 'w') as _f:
                json.dump({'epoch': epoch,
                           'val_miss': float(best_val_miss),
                           'checkpoint': f'rolling_ep{epoch}_end.pt'}, _f, indent=1)
            print(f"  ** New best val_miss={best_val_miss:.4f} "
                  f"(best.json -> rolling_ep{epoch}_end.pt)")
            # ---- SUPERSEDED 2026-08-15 ----
            # Checkpoint pruning disabled 2026-08-15: it deleted all but the last 2 best_*.pt
            # # (and all but 1-3 rolling_*.pt), which is why earlier runs lost their early-epoch
            # # checkpoints. Seed-comparison runs need every epoch retained.
            # import glob as _glob
            # bests = sorted(_glob.glob(os.path.join(args.save_dir, 'best_*.pt')),
            # key=os.path.getmtime)
            # while len(bests) > 2:
            # os.remove(bests.pop(0))
            # ---- end superseded ----


if __name__ == '__main__':
    main()
