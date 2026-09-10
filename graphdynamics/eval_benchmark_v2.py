#!/usr/bin/env python3
"""
Benchmark evaluation for RecurrentBrainNet 256x256 on CelebA-HQ.

Generates irregular masks in 6 coverage buckets (Liu et al. 2018 convention),
computes L1 (%), PSNR, SSIM, LPIPS per bucket, and compares against published
baselines (PConv, EdgeConnect, RFR, LaMa, MAT).

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 eval_benchmark.py --checkpoint checkpoints_256/best_ep3.pt
    python3 eval_benchmark.py --checkpoint checkpoints_256/latest.pt --save results_latest.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ---------------------------------------------------------------------------
# Metric imports
# ---------------------------------------------------------------------------
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

try:
    import lpips as lpips_pkg
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[WARN] lpips package not found — LPIPS metric will be skipped.")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Legacy architectures (v2/v4/v5/v6/v7/loop3/v7f2000/v10super/v1000super/v1001super)
# were removed from this package on 2026-09-07. Only the three architectures the
# paper reports remain: unet, v1003super, feeder1004. Each imports itself locally
# in load_model() below.


# ============================================================================
# Irregular mask generation (adapted from data_256.py)
# ============================================================================

# ---- SUPERSEDED 2026-08-15 ----------------------------------------------
# Original generator: fixed 1-4 rects + 3-8 thick strokes, no coverage control.
# Over 3000 seeds it produced min 16.5% coverage and exactly 1 sample in the
# 10-20% bucket, so the reported '0-20%' column was 20 forced outliers at 17.9%
# mean coverage. Replaced by a coverage-targeted version below.
# Kept for reference; do not delete.
# def generate_irregular_mask(h, w, rng):
#     """
#     Generate an irregular mask using random brush strokes + rectangles.
#     Returns numpy array (h, w) with 1=observed, 0=hidden.
#     No target ratio clamping — we want the natural distribution so we can
#     bucket by actual coverage.
#     """
#     mask = np.ones((h, w), dtype=np.float32)
#
#     # 1-4 random rectangles
#     n_rect = rng.integers(1, 5)
#     for _ in range(n_rect):
#         rect_h = rng.integers(h // 8, h // 2)
#         rect_w = rng.integers(w // 8, w // 2)
#         y0 = rng.integers(0, max(1, h - rect_h))
#         x0 = rng.integers(0, max(1, w - rect_w))
#         mask[y0:y0 + rect_h, x0:x0 + rect_w] = 0.0
#
#     # 3-8 random brush strokes
#     n_strokes = rng.integers(3, 9)
#     for _ in range(n_strokes):
#         y, x = rng.integers(0, h), rng.integers(0, w)
#         thickness = rng.integers(6, 30)
#         n_steps = rng.integers(15, 80)
#         yy, xx = np.ogrid[-thickness:thickness + 1, -thickness:thickness + 1]
#         circle = yy ** 2 + xx ** 2 <= thickness ** 2
#         for _ in range(n_steps):
#             y_start = max(0, y - thickness)
#             x_start = max(0, x - thickness)
#             y_end = min(h, y + thickness + 1)
#             x_end = min(w, x + thickness + 1)
#             cy_start = max(0, thickness - y)
#             cx_start = max(0, thickness - x)
#             cy_end = cy_start + (y_end - y_start)
#             cx_end = cx_start + (x_end - x_start)
#             mask[y_start:y_end, x_start:x_end][
#                 circle[cy_start:cy_end, cx_start:cx_end]] = 0.0
#             angle = rng.uniform(0, 2 * np.pi)
#             step = rng.integers(5, 20)
#             y = int(np.clip(y + step * np.sin(angle), 0, h - 1))
#             x = int(np.clip(x + step * np.cos(angle), 0, w - 1))
#
#     return mask
# ---- end superseded ------------------------------------------------------

# ---- MOVED 2026-08-16 ----
# generate_irregular_mask now lives in data_256.py and is shared by training,
# validation and this benchmark, so all three draw masks with identical geometry
# and the ONLY difference between them is the coverage target. Previously the
# dataset had its own generator whose overshoot corrector deleted individual
# pixels, shattering the holes -- models were trained and selected on confetti
# masks and then benchmarked on real ones. The body that used to be here is
# unchanged; it simply lives in data_256.py now.
from data_256 import generate_irregular_mask   # noqa: E402



# ---- ADDED 2026-09-09: evaluate on the TRAINING mask distribution ----
# The benchmark called generate_irregular_mask() with no target, so the
# generator used its own default band, U(0.02, 0.70).  Training draws its
# target from a band around mask_ratio -- U(0.09, 0.49) at the default 0.25.
#
# ---- CORRECTED 2026-09-10 ----
# This note used to say the training band gives achieved coverage in
# [0.093, 0.541] and that 22% of scored samples lay outside it, 29% of the
# 40-60% band being ABOVE it.  Both were wrong: they came from too small a
# training sample.  Re-measured over 30,000 training draws, achieved coverage
# runs [0.0904, 0.6040] -- the generator overshoots its target, so the heavy
# end of training reaches past the heaviest scored mask (0.5999).  There is
# no upper gap.  Against the real 4,893-sample scored set the true figure is
# 11.8% outside (579 samples), ALL of them below the training floor and all
# in the 0-20% band, which is 38.6% of that band; the 20-40% and 40-60%
# bands are entirely inside.  Stable across thresholds: 11.8% at the
# measured training minimum, 12.2% at its 0.1st percentile, 11.8% at the
# 0.09 target floor.
#
# --mask-dist training draws the target from exactly the band data_256 uses,
# so the evaluation distribution matches the training one.  One uniform is
# consumed from the same rng either way, so the only thing that changes is
# the range it is drawn from.  The default stays 'benchmark', so existing
# results reproduce unchanged.
def _draw_mask(h, w, rng, mask_dist, mask_ratio):
    if mask_dist == 'training':
        lo = max(0.09, mask_ratio - 0.16)
        hi = min(0.8, mask_ratio + 0.24)
        return generate_irregular_mask(h, w, rng, target=rng.uniform(lo, hi))
    return generate_irregular_mask(h, w, rng)


def mask_coverage(mask_np):
    """Return fraction of pixels that are hidden (0)."""
    return 1.0 - mask_np.mean()


# Bucket boundaries: [lo, hi) for each of 6 buckets
BUCKET_BOUNDS = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
]
BUCKET_NAMES = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%"
]
NUM_BUCKETS = len(BUCKET_BOUNDS)
MIN_PER_BUCKET = 20


def assign_bucket(coverage):
    """Return bucket index for a given mask coverage, or -1 if out of range."""
    for i, (lo, hi) in enumerate(BUCKET_BOUNDS):
        if lo <= coverage < hi:
            return i
    return -1


def generate_bucketed_masks(n_images, h, w, min_per_bucket=MIN_PER_BUCKET,
                            mask_dist='benchmark', mask_ratio=0.25):
    """
    Generate masks for each image assigned to buckets.

    For each image, try the canonical seed first (image_idx * 6 + bucket_idx).
    If a bucket is underfilled, keep generating with increasing seed offsets
    until every bucket has at least min_per_bucket samples.

    Returns:
        bucket_data: list of NUM_BUCKETS lists, each containing
                     (image_idx, mask_np) tuples.
    """
    bucket_data = [[] for _ in range(NUM_BUCKETS)]

    # Phase 1: canonical seeds — one attempt per image per bucket
    print(f"Generating masks: phase 1 (canonical seeds, {n_images} images x {NUM_BUCKETS} buckets)...")
    for img_idx in range(n_images):
        for bkt in range(NUM_BUCKETS):
            seed = img_idx * NUM_BUCKETS + bkt
            rng = np.random.default_rng(seed)
            mask = _draw_mask(h, w, rng, mask_dist, mask_ratio)
            cov = mask_coverage(mask)
            actual_bkt = assign_bucket(cov)
            if actual_bkt >= 0:
                bucket_data[actual_bkt].append((img_idx, mask, seed))

    # Report phase 1 counts
    for i, name in enumerate(BUCKET_NAMES):
        print(f"  {name}: {len(bucket_data[i])} samples")

    # Phase 2: fill underpopulated buckets with extra seeds
    extra_seed_base = n_images * NUM_BUCKETS
    max_extra_attempts = n_images * 10  # cap attempts to avoid spinning

    for bkt_idx in range(NUM_BUCKETS):
        lo, hi = BUCKET_BOUNDS[bkt_idx]
        target_mid = (lo + hi) / 2.0
        attempt = 0
        while len(bucket_data[bkt_idx]) < min_per_bucket and attempt < max_extra_attempts:
            # Cycle through images to spread across different faces
            img_idx = attempt % n_images
            seed = extra_seed_base + bkt_idx * max_extra_attempts + attempt
            rng = np.random.default_rng(seed)
            mask = _draw_mask(h, w, rng, mask_dist, mask_ratio)
            cov = mask_coverage(mask)
            actual_bkt = assign_bucket(cov)
            if actual_bkt == bkt_idx:
                bucket_data[bkt_idx].append((img_idx, mask, seed))
            attempt += 1

        if len(bucket_data[bkt_idx]) < min_per_bucket:
            print(f"  [WARN] {BUCKET_NAMES[bkt_idx]}: only {len(bucket_data[bkt_idx])} "
                  f"samples (target {min_per_bucket})")

    # Final counts
    print("Final bucket counts:")
    for i, name in enumerate(BUCKET_NAMES):
        print(f"  {name}: {len(bucket_data[i])} samples")

    return bucket_data


# ============================================================================
# Metrics
# ============================================================================

def compute_l1_percent(pred, gt):
    """Full-image L1 x 100 (what papers report)."""
    return (pred - gt).abs().mean().item() * 100.0


def compute_psnr(pred_np, gt_np):
    """PSNR in dB. Images in [0, 1]."""
    return psnr_fn(gt_np, pred_np, data_range=1.0)


def compute_ssim(pred_np, gt_np):
    """SSIM. Images in [0, 1], channel_axis=0 for (C, H, W)."""
    return ssim_fn(gt_np, pred_np, data_range=1.0, channel_axis=0)


# ============================================================================
# Main evaluation
# ============================================================================

def load_model(checkpoint_path, graph_path, device,
               v1003super=False, unet=False, feeder1004=False, **_removed):
    """Load a model from a checkpoint."""
    # ---- ADDED 2026-08-16: the two arms currently training ----
    # Neither had a branch here. Running this script on either checkpoint fell
    # through to the RecurrentBrainNet default below and died on a strict
    # load_state_dict -- i.e. two 36-52h runs with no way to score them. The only
    # script that could build a UNetOnly was eval_benchmark_unet.py (April), which
    # imports the PRE-FIX mask generator and scores all 2000 validation images,
    # selection half included. Do not use it.
    # ---- ADDED 2026-08-29: feeder1004 (v1003 minus supernode attention) ----
    # This script was last touched 2026-08-16; feeder1004.py was created
    # 2026-08-26, so the 3-seed ablation had no branch here and fell through to
    # the RecurrentBrainNet default, dying on the strict load. Its class is also
    # named RecurrentBrainNetV7 (same as v1003super's), so it needs its own
    # import off brain_v7/, exactly like the v1003super branch below.
    # Its checkpoints carry NO attn_* keys and the class defines none, so the
    # strict=True load below is the check that the right class was built.
    if feeder1004:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'feeder1004'))
        from feeder1004 import RecurrentBrainNetV7 as _F1004
        model = _F1004(graph_path, brain_channels=32, n_iters=5)
    elif unet:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'unet'))
        from model_unetonly import UNetOnly
        model = UNetOnly()
    elif v1003super:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'v1003'))
        from model_v1003super import RecurrentBrainNetV7 as _V1003
        model = _V1003(graph_path, brain_channels=32, n_iters=5)
    else:
        raise SystemExit(
            "Specify the architecture: --unet, --v1003super, or --feeder1004.\n"
            "Older architectures were removed from this package on 2026-09-07.")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        state = ckpt['model']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    # Strip "module." prefix from DDP
    clean = {}
    for k, v in state.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        clean[key] = v

    model.load_state_dict(clean, strict=True)
    model = model.to(device)
    model.eval()
    return model


def load_val_images(val_dir):
    """Load all validation image paths, sorted."""
    import glob
    files = sorted(glob.glob(os.path.join(val_dir, '*.jpg')))
    if not files:
        files = sorted(glob.glob(os.path.join(val_dir, '*.png')))
    if not files:
        raise FileNotFoundError(f"No images found in {val_dir}")
    return files


def load_and_preprocess(path, transform):
    """Load a single image and apply transforms. Returns (3, 256, 256) tensor in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return transform(img)


@torch.no_grad()
def run_evaluation(model, val_files, bucket_data, device, lpips_model=None):
    """
    Run model inference on all (image, mask) pairs in each bucket.
    Returns per-bucket metric dictionaries.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    # Pre-load all validation images into memory for efficiency
    print(f"Loading {len(val_files)} validation images...")
    val_images = []
    for f in val_files:
        val_images.append(load_and_preprocess(f, transform))
    print("Images loaded.")

    # Accumulate metrics per bucket
    results = []
    for bkt_idx in range(NUM_BUCKETS):
        results.append({
            'l1': [], 'psnr': [], 'ssim': [], 'lpips': [],
            # ADDED 2026-08-16: hidden-region L1, i.e. the SAME quantity the
            # training loop calls val_miss. Without it the benchmark headline
            # (full-image L1 x100) and the training headline (hidden-region L1)
            # are two different numbers that cannot be reconciled.
            'l1_hidden': [],
        })

    # ADDED 2026-08-16: flat per-sample dump. Bucket means alone cannot support a
    # paired test between two arms; the pairing key is (bucket, img_idx, seed),
    # which is deterministic given the same val_files list, so the same rows
    # appear for both models and can be differenced sample by sample.
    per_sample = []

    total_samples = sum(len(bd) for bd in bucket_data)
    processed = 0
    t0 = time.time()

    for bkt_idx in range(NUM_BUCKETS):
        samples = bucket_data[bkt_idx]
        if not samples:
            continue

        for img_idx, mask_np, mask_seed in samples:
            gt = val_images[img_idx]  # (3, 256, 256) in [0, 1]

            # Prepare mask tensor: (1, 256, 256), 1=observed, 0=hidden
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # (1, 256, 256)

            # Prepare input
            x_obs = gt * mask_t  # (3, 256, 256)

            # Batch dimension
            x_obs_b = x_obs.unsqueeze(0).to(device)  # (1, 3, 256, 256)
            mask_b = mask_t.unsqueeze(0).to(device)   # (1, 1, 256, 256)
            gt_b = gt.unsqueeze(0).to(device)         # (1, 3, 256, 256)

            # Forward
            recons, _ = model(x_obs_b, mask_b)
            pred = recons[0]  # (1, 3, 256, 256)
            pred = pred.clamp(0, 1)

            # L1 (%) — full image
            l1_val = compute_l1_percent(pred[0].cpu(), gt)
            results[bkt_idx]['l1'].append(l1_val)

            # Hidden-region L1 (training's val_miss): mean |pred - gt| over the
            # holes only, all 3 channels.
            hid = (1.0 - mask_t)
            n_hid = hid.sum() * 3
            l1_hidden = float(((pred[0].cpu() - gt).abs() * hid).sum() / n_hid.clamp(min=1))
            results[bkt_idx]['l1_hidden'].append(l1_hidden)

            # PSNR, SSIM — on numpy
            pred_np = pred[0].cpu().numpy()
            gt_np = gt.numpy()
            results[bkt_idx]['psnr'].append(compute_psnr(pred_np, gt_np))
            results[bkt_idx]['ssim'].append(compute_ssim(pred_np, gt_np))

            # LPIPS
            if lpips_model is not None:
                with torch.amp.autocast('cuda', enabled=False):
                    # LPIPS expects [-1, 1]
                    lpips_val = lpips_model(
                        pred.float() * 2 - 1,
                        gt_b.float() * 2 - 1
                    ).item()
                results[bkt_idx]['lpips'].append(lpips_val)

            per_sample.append({
                'bucket': BUCKET_NAMES[bkt_idx],
                'bucket_idx': bkt_idx,
                'img_idx': int(img_idx),
                'seed': int(mask_seed),
                'mask_frac_hidden': float(1.0 - mask_np.mean()),
                'l1': float(l1_val),
                'l1_hidden': float(l1_hidden),
                'psnr': float(results[bkt_idx]['psnr'][-1]),
                'ssim': float(results[bkt_idx]['ssim'][-1]),
                'lpips': (float(results[bkt_idx]['lpips'][-1])
                          if results[bkt_idx]['lpips'] else None),
            })

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (total_samples - processed) / rate if rate > 0 else 0
                print(f"  [{processed}/{total_samples}] "
                      f"{rate:.1f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Inference complete: {processed} samples in {elapsed:.1f}s "
          f"({processed / elapsed:.1f} samples/s)")

    return results, per_sample


def aggregate_results(results):
    """Compute mean and std for each metric in each bucket."""
    summary = []
    for bkt_idx in range(NUM_BUCKETS):
        bkt = results[bkt_idx]
        n = len(bkt['l1'])
        entry = {
            'bucket': BUCKET_NAMES[bkt_idx],
            'n_samples': n,
        }
        if n > 0:
            for metric in ['l1', 'l1_hidden', 'psnr', 'ssim', 'lpips']:
                vals = bkt[metric]
                if vals:
                    entry[f'{metric}_mean'] = float(np.mean(vals))
                    entry[f'{metric}_std'] = float(np.std(vals))
                else:
                    entry[f'{metric}_mean'] = None
                    entry[f'{metric}_std'] = None
        else:
            for metric in ['l1', 'l1_hidden', 'psnr', 'ssim', 'lpips']:
                entry[f'{metric}_mean'] = None
                entry[f'{metric}_std'] = None
        summary.append(entry)
    return summary


# ============================================================================
# Published baselines (approximate, from papers, CelebA-HQ 256x256 irregular)
# ============================================================================
# L1 values are L1 (%) = full-image L1 x 100
# Grouped as 0-20%, 20-40%, 40-60% (papers often merge adjacent buckets)

# ---- SUPERSEDED 2026-08-15 ----------------------------------------------
# Published-baseline table was unsourced and partly impossible: PConv reports
# Places2 512x512 only, LaMa and MAT publish no L1 at all, RFR is CVPR'20 not
# ECCV'20 and reports CelebA at deciles. Never appeared in main.tex.
# Kept for reference; do not delete.
# BASELINES = [
#     {'method': 'PConv',       'venue': "ECCV'18",   'l1_0_20': 0.95, 'l1_20_40': 2.67, 'l1_40_60': 5.30, 'fid': None},
#     {'method': 'EdgeConnect', 'venue': "ICCVW'19",  'l1_0_20': 0.91, 'l1_20_40': 2.35, 'l1_40_60': 4.50, 'fid': None},
#     {'method': 'RFR',         'venue': "ECCV'20",   'l1_0_20': 0.82, 'l1_20_40': 2.12, 'l1_40_60': 4.21, 'fid': None},
#     {'method': 'LaMa',        'venue': "WACV'22",   'l1_0_20': 0.60, 'l1_20_40': 1.63, 'l1_40_60': 3.53, 'fid': 5.2},
#     {'method': 'MAT',         'venue': "CVPR'22",   'l1_0_20': 0.54, 'l1_20_40': 1.48, 'l1_40_60': 3.18, 'fid': 4.5},
# ]
# ---- end superseded ------------------------------------------------------

BASELINES = []   # emptied 2026-08-15, see superseded block above


def print_results_table(summary, checkpoint_name):
    """Print a formatted comparison table with published baselines."""

    # Compute merged L1 for our model: average adjacent bucket means
    def merged_l1(summary, bkt_indices):
        # weight bucket means by sample count (was an unweighted mean of means)
        rows = [summary[i] for i in bkt_indices
                if summary[i]['l1_mean'] is not None and summary[i]['n_samples']]
        if not rows:
            return None
        n = sum(r['n_samples'] for r in rows)
        return sum(r['l1_mean'] * r['n_samples'] for r in rows) / n

    our_l1_0_20 = merged_l1(summary, [0, 1])
    our_l1_20_40 = merged_l1(summary, [2, 3])
    our_l1_40_60 = merged_l1(summary, [4, 5])

    sep = "-" * 82
    print()
    print("=" * 82)
    print("  INPAINTING BENCHMARK: CelebA-HQ 256x256, Irregular Masks (Liu et al. 2018)")
    print("=" * 82)

    # --- Per-bucket detailed table ---
    print()
    print("Per-bucket results (ours):")
    print(sep)
    print(f"{'Bucket':>8s}  {'N':>5s}  {'L1(%)':>7s}  {'PSNR(dB)':>9s}  "
          f"{'SSIM':>7s}  {'LPIPS':>7s}")
    print(sep)
    for entry in summary:
        n = entry['n_samples']
        l1 = f"{entry['l1_mean']:.3f}" if entry['l1_mean'] is not None else "  —"
        psnr = f"{entry['psnr_mean']:.2f}" if entry['psnr_mean'] is not None else "   —"
        ssim_v = f"{entry['ssim_mean']:.4f}" if entry['ssim_mean'] is not None else "  —"
        lpips_v = f"{entry['lpips_mean']:.4f}" if entry['lpips_mean'] is not None else "  —"
        print(f"{entry['bucket']:>8s}  {n:>5d}  {l1:>7s}  {psnr:>9s}  "
              f"{ssim_v:>7s}  {lpips_v:>7s}")
    print(sep)

    # --- Comparison table with baselines ---
    print()
    print("Comparison with published baselines (L1 %, merged buckets):")
    print(sep)
    print(f"{'Method':>16s}  {'Venue':>10s}  {'0-20%':>7s}  {'20-40%':>8s}  "
          f"{'40-60%':>8s}  {'FID':>6s}")
    print(sep)

    for b in BASELINES:
        fid_str = f"{b['fid']:.1f}" if b['fid'] is not None else "  —"
        print(f"{b['method']:>16s}  {b['venue']:>10s}  {b['l1_0_20']:>7.2f}  "
              f"{b['l1_20_40']:>8.2f}  {b['l1_40_60']:>8.2f}  {fid_str:>6s}")

    # Our model row
    def fmt_l1(v):
        return f"{v:.2f}" if v is not None else "  —"

    ckpt_label = f"Ours ({checkpoint_name})"
    print(sep)
    print(f"{ckpt_label:>16s}  {'':>10s}  {fmt_l1(our_l1_0_20):>7s}  "
          f"{fmt_l1(our_l1_20_40):>8s}  {fmt_l1(our_l1_40_60):>8s}  {'  —':>6s}")
    print(sep)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RecurrentBrainNet on CelebA-HQ 256 irregular masks")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument('--graph', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_brain_mild_v146_feeder_supernode.npz'),
                        help="Path to graph .npz file")
    # ---- ADDED 2026-09-09: --data, and no hardcoded dataset path ----
    # --val-dir defaulted to an absolute path on the author's machine, and the
    # README documented `--data` -- which this parser did not accept, so every
    # documented evaluation command died on "unrecognized arguments: --data".
    # --data is now the primary flag and takes the dataset ROOT, matching the
    # three trainers; --val-dir survives as an override for a loose directory
    # of images. With neither, celebahq.resolve_val_dir() falls back to
    # $TENSORMIND_CELEBAHQ, then to a cached copy, then downloads one.
    parser.add_argument('--data', type=str, default=None,
                        help="CelebA-HQ 256 root, holding train/ and validation/. "
                             "Omit to use $TENSORMIND_CELEBAHQ, a cached copy, or "
                             "to download a pinned one.")
    parser.add_argument('--val-dir', type=str, default=None,
                        help="Validation image directory. Overrides --data; normally "
                             "you want --data.")
    parser.add_argument('--no-download', action='store_true',
                        help="Never download the dataset; fail instead.")
    parser.add_argument('--mask-dist', choices=['benchmark', 'training'],
                        default='benchmark',
                        help="Coverage band the evaluation masks are drawn from. "
                             "'benchmark' (default) is the published U(0.02,0.70); "
                             "'training' matches what the models were trained on.")
    parser.add_argument('--mask-ratio', type=float, default=0.25,
                        help="Centre of the training coverage band, for --mask-dist "
                             "training. Must match the value the model was trained with.")
    # ---- ADDED 2026-08-15 (issue 11) ----
    # There was no held-out test set: the benchmark scored the same 2000 images used
    # to select best_ep*.pt. val_test_split_v1.json splits them 1000/1000 so the
    # reported metric is not the one that was optimised over.
    parser.add_argument('--v1003super', action='store_true',
                        help='graph arm currently in training (75,982,503 params)')
    parser.add_argument('--unet', action='store_true',
                        help='UNet baseline arm (87,103,619 params)')
    parser.add_argument('--val-subset', type=str, default='test',
                        choices=['all', 'selection', 'test'],
                        help="which half of the validation set to score (default: test)")
    parser.add_argument('--split-file', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'val_test_split_v1.json'))
    parser.add_argument('--save', type=str, default=None,
                        help="Path to save JSON results (default: auto from checkpoint name)")
    parser.add_argument('--min-per-bucket', type=int, default=MIN_PER_BUCKET,
                        help="Minimum samples per mask-coverage bucket")
    parser.add_argument('--device', type=str, default=None,
                        help="Device (cuda / cpu). Auto-detect if omitted.")
    parser.add_argument('--v2', action='store_true',
                        help="Use RecurrentBrainNetV2 (multi-channel brain)")
    parser.add_argument('--v5', action='store_true',
                        help='Use V5 brain (LN before feedback)')
    parser.add_argument('--v4', action='store_true',
                        help="Use RecurrentBrainNetV4 (LN on a1, feedback init=1)")
    parser.add_argument('--v6', action='store_true',
                        help='Use V6 brain (short feedback layer_32)')
    parser.add_argument('--v7', action='store_true',
                        help='Use V7 brain (deep multi-scale, 5 iters)')
    parser.add_argument('--loop3', action='store_true',
                        help='Use V7 loop3 brain (3-layer recurrent)')
    parser.add_argument('--v7f2000', action='store_true',
                        help='Use V7feeder2000 (no-delay, no attention)')
    parser.add_argument('--v10super', action='store_true',
                        help='Use V10 supernode (no-delay, shared QKV attention)')
    parser.add_argument('--v1001super', action='store_true',
                        help='Use V1001 supernode (delay, classless attention)')
    parser.add_argument('--v1000super', action='store_true',
                        help='Use V1000 supernode (delay, size-class attention)')
    parser.add_argument('--feeder1004', action='store_true',
                        help='Use feeder1004 (v1003 minus supernode attention)')
    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(script_dir, args.checkpoint)
    if not os.path.isabs(args.graph):
        args.graph = os.path.join(script_dir, args.graph)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.graph):
        print(f"ERROR: graph not found: {args.graph}")
        sys.exit(1)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.graph, device,
                       v1003super=getattr(args, 'v1003super', False),
                       unet=getattr(args, 'unet', False),
                       feeder1004=getattr(args, 'feeder1004', False))

    # Optional: LPIPS model (VGG backbone, fp32 — known fp16 overflow issue)
    lpips_model = None
    if HAS_LPIPS:
        print("Loading LPIPS model (VGG)...")
        lpips_model = lpips_pkg.LPIPS(net='vgg').to(device)
        lpips_model.eval()

    # Load validation image paths
    # ---- ADDED 2026-09-09 ---- resolve the dataset, downloading if needed.
    from celebahq import resolve_val_dir
    args.val_dir = resolve_val_dir(args.val_dir, args.data,
                                   download=not args.no_download)
    print(f"Validation directory: {args.val_dir}")
    val_files = load_val_images(args.val_dir)
    # ---- ADDED 2026-08-15 (issue 11) ----
    # Restrict to one half of the validation set so the reported metric is not the
    # one used for checkpoint selection. Default 'test'. Use --val-subset all to
    # reproduce pre-2026-08-15 numbers.
    if args.val_subset != 'all':
        import json as _json
        _sp = _json.load(open(args.split_file))
        _idx = _sp['selection_idx'] if args.val_subset == 'selection' else _sp['test_idx']
        if len(val_files) != _sp['n_total']:
            raise ValueError(f"split file expects {_sp['n_total']} images, found {len(val_files)}")
        val_files = [val_files[i] for i in _idx]
    n_images = len(val_files)
    print(f"Validation images: {n_images}  (subset={args.val_subset})")

    # Generate bucketed masks
    print(f"Mask distribution: {args.mask_dist}"
          + (f" (band around mask_ratio={args.mask_ratio})" if args.mask_dist == 'training' else ""))
    bucket_data = generate_bucketed_masks(
        n_images, 256, 256, min_per_bucket=args.min_per_bucket,
        mask_dist=args.mask_dist, mask_ratio=args.mask_ratio)

    # Run evaluation
    print("\nRunning inference...")
    results, per_sample = run_evaluation(model, val_files, bucket_data, device, lpips_model)

    # Aggregate
    summary = aggregate_results(results)

    # Print table
    ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
    print_results_table(summary, ckpt_name)

    # Save JSON
    if args.save is None:
        # ---- CHANGED 2026-08-16: include the run directory in the name ----
        # Both arms' final checkpoints are named rolling_ep49_end.pt, in
        # different directories, so the old name collided and scoring the second
        # arm silently overwrote the first arm's JSON -- including the
        # per_sample dump the paired test depends on.
        # was: args.save = os.path.join(script_dir, f"eval_results_{ckpt_name}.json")
        run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        args.save = os.path.join(
            script_dir, f"eval_results_{run_name}_{ckpt_name}.json")

    output = {
        'checkpoint': args.checkpoint,
        'graph': args.graph,
        'val_dir': args.val_dir,
        'n_val_images': n_images,
        'mask_dist': args.mask_dist,
        'mask_ratio': args.mask_ratio,
        'device': str(device),
        'buckets': summary,
        'baselines': BASELINES,
        # ADDED 2026-08-16: every sample, so an arm-vs-arm paired test is
        # possible after the fact. Pairing key: (bucket_idx, img_idx, seed).
        'per_sample': per_sample,
    }
    with open(args.save, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.save}")


if __name__ == '__main__':
    main()
