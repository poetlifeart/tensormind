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
    ap.add_argument('--val-dir', default='/home/vahid/data/celebahq256/validation/')
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

    val_files = EB.load_val_images(a.val_dir)
    sp = json.load(open(a.split_file))
    val_files = [val_files[i] for i in sp['test_idx']]
    print(f"test images: {len(val_files)}", flush=True)

    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                             transforms.ToTensor()])
    imgs = [EB.load_and_preprocess(f, tf) for f in val_files]
    bucket_data = EB.generate_bucketed_masks(len(val_files), 256, 256)

    BANDS = {'0-20%': [0, 1], '20-40%': [2, 3], '40-60%': [4, 5]}
    fids = {}
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
        print(f"  {band:>7}  FID {fids[band]:8.3f}   "
              f"({len(seen_real)} real, {time.time()-t0:.0f}s)", flush=True)

    rec = dict(checkpoint=a.checkpoint, fid=fids, n_real=len(val_files))
    if a.save:
        json.dump(rec, open(a.save, 'w'), indent=1)
        print(f"  -> {a.save}")


if __name__ == '__main__':
    main()
