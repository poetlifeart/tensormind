#!/usr/bin/env python3
"""FID for the three-arm inpainting benchmark.

Reuses eval_benchmark_v2's model loader, validation split and mask generator,
so the images and masks are byte-identical to the ones already scored -- same
canonical seeds (img_idx * 6 + bucket), same held-out test half.  Only the
metric differs.

FID is computed on the COMPOSITED output (observed pixels preserved, hole
filled), against the real test images, using torchmetrics' Inception-v3
features.  Reported pooled over all masks and per coverage band.

CAVEAT worth stating in the paper: the held-out half has 1,000 images, below
the 2,048 feature dimensions of the Inception pool layer, so the covariance
estimate is rank-deficient and the absolute FID is biased upward.  The bias is
identical across arms -- same reals, same masks -- so the comparison stands
while the absolute values should not be read against published numbers.

The 1,000 applies to the pooled row only.  Each BAND draws its reference set
from just the images that received a mask in that coverage range: 828, 861 and
849 on the published run, so the per-band rows are more rank-deficient still,
and the three bands do not share a reference set with each other.  Comparisons
BETWEEN ARMS within a band remain exact -- identical reals, identical masks.
`n_real` in the output records the per-band counts (it used to record 1,000
for every row, which is where the paper's figure came from).

Usage:
  python3 eval_fid_2026-08-31.py --unet --checkpoint <ckpt> --save fid_x.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# original (pre-tensormind): '/home/vahid/experimentbrain'
import eval_benchmark_v2 as EB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--graph', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_brain_mild_v146_feeder_supernode.npz'))
    # ---- ADDED 2026-09-09 ---- see the note in eval_benchmark_v2.py: --data is
    # the documented flag, --val-dir is the override, and neither being given
    # resolves via $TENSORMIND_CELEBAHQ / cache / pinned download.
    ap.add_argument('--data', default=None,
                    help="CelebA-HQ 256 root, holding train/ and validation/.")
    ap.add_argument('--val-dir', default=None,
                    help="Validation image directory. Overrides --data.")
    ap.add_argument('--no-download', action='store_true',
                    help="Never download the dataset; fail instead.")
    ap.add_argument('--mask-dist', choices=['benchmark', 'training'],
                    default='benchmark',
                    help="Coverage band for the evaluation masks. 'benchmark' "
                         "(default) is the published U(0.02,0.70); 'training' "
                         "matches what the models were trained on.")
    ap.add_argument('--mask-ratio', type=float, default=0.25,
                    help="Centre of the training coverage band, for --mask-dist training.")
    ap.add_argument('--split-file', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'val_test_split_v1.json'))
    ap.add_argument('--unet', action='store_true')
    ap.add_argument('--v1003super', action='store_true')
    ap.add_argument('--feeder1004', action='store_true')
    ap.add_argument('--save', default=None)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()

    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    from torchmetrics.image.fid import FrechetInceptionDistance

    model = EB.load_model(a.checkpoint, a.graph, dev, unet=a.unet,
                          v1003super=a.v1003super, feeder1004=a.feeder1004)
    model.eval()

    from celebahq import resolve_val_dir
    a.val_dir = resolve_val_dir(a.val_dir, a.data, download=not a.no_download)
    print(f"validation directory: {a.val_dir}", flush=True)
    val_files = EB.load_val_images(a.val_dir)
    sp = json.load(open(a.split_file))
    val_files = [val_files[i] for i in sp['test_idx']]
    print(f"test images: {len(val_files)}", flush=True)

    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                             transforms.ToTensor()])
    imgs = [EB.load_and_preprocess(f, tf) for f in val_files]
    # ---- ADDED 2026-09-09 ---- same --mask-dist option as the benchmark, so
    # FID can be computed on the coverage band the models were trained on
    # rather than the wider published one. Default unchanged.
    print(f"mask distribution: {a.mask_dist}", flush=True)
    bucket_data = EB.generate_bucketed_masks(len(val_files), 256, 256,
                                             mask_dist=a.mask_dist,
                                             mask_ratio=a.mask_ratio)

    BANDS = {'0-20%': [0, 1], '20-40%': [2, 3], '40-60%': [4, 5]}
    fids, n_reals = {}, {}
    t0 = time.time()
    for band, bkts in list(BANDS.items()) + [('all', list(range(6)))]:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
        seen_real = set()
        with torch.no_grad():
            for b in bkts:
                for img_idx, mask_np, _seed in bucket_data[b]:
                    # exactly the benchmark's call: model(x_obs, mask) -> (recons, _)
                    # and recons[0] is already the composited reconstruction.
                    gt = imgs[img_idx]
                    m_t = torch.from_numpy(mask_np).float().unsqueeze(0)
                    x = gt.unsqueeze(0).to(dev)
                    x_obs = (gt * m_t).unsqueeze(0).to(dev)
                    m = m_t.unsqueeze(0).to(dev)
                    recons, _ = model(x_obs, m)
                    comp = recons[0].clamp(0, 1)
                    fid.update(comp, real=False)
                    if img_idx not in seen_real:
                        fid.update(x.clamp(0, 1), real=True)
                        seen_real.add(img_idx)
        fids[band] = float(fid.compute())
        n_reals[band] = len(seen_real)
        print(f"  {band:>7}  FID {fids[band]:8.3f}   "
              f"({len(seen_real)} real, {time.time()-t0:.0f}s)", flush=True)

    # ---- FIXED 2026-09-09 ----
    # was: n_real=len(val_files) -- recorded 1,000 for every band, but only the
    # pooled row actually draws on all 1,000. Each band's reference set is just
    # the images that received a mask in its coverage range (828-861 measured on
    # the published run), so the per-band rows are more rank-deficient than the
    # figure suggested, and the paper's caveat inherited the wrong number.
    rec = dict(checkpoint=a.checkpoint, fid=fids, n_real=n_reals,
               n_test_images=len(val_files), mask_dist=a.mask_dist)
    if a.save:
        json.dump(rec, open(a.save, 'w'), indent=1)
        print(f"  -> {a.save}")


if __name__ == '__main__':
    main()
