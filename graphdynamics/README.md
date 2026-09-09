# graphdynamics

The three inpainting models from the paper's benchmark table, their training
scripts, and the two evaluation scripts.

| paper row | model | trainer |
|---|---|---|
| Graph${+}$attn. | `v1003/model_v1003super.py` | `v1003/train_v1003super.py` |
| Graph | `feeder1004/feeder1004.py` | `feeder1004/train_feeder1004.py` |
| U-Net | `unet/model_unetonly.py` | `unet/train_unetonly_rolling.py` |

`feeder1004` is `v1003super` with the supernode attention removed — the ablation.
Its header records the parent's md5 and the three lines that differ.

## What ships here

```
graph_brain_mild_v146_feeder_supernode.npz   16,807 nodes / 344,483 edges, 1,015 supernodes
val_test_split_v1.json                       which 1,000 validation images are the reporting half
celebahq.py                                   dataset resolution + optional download
celebahq_val_manifest_v1.npz                  fingerprints of the 2,000 validation images, in order
data_256.py                                   dataset + irregular mask generation
eval_benchmark_v2.py                          PSNR / SSIM / LPIPS / L1 by mask coverage
eval_fid_2026-08-31.py                        FID
```

The graph and the split are the defaults for `--graph` and `--split-file`, so you
do not need to pass either. Both resolve relative to the script, so any working
directory works.

## The data

**CelebA-HQ 256x256.** You do not have to supply it: with no `--data`, every
script here fetches a pinned copy on first use and reuses it afterwards.

```bash
# nothing to do -- downloads ~3 GB to ~/.cache/tensormind/celebahq256, once
python3 v1003/train_v1003super.py --save-dir ckpt_v1003_s0

# or point at your own copy
python3 v1003/train_v1003super.py --data /path/to/celebahq256 --save-dir ckpt_v1003_s0

# or set it once for every script
export TENSORMIND_CELEBAHQ=/path/to/celebahq256

# or refuse to download and fail loudly instead
python3 v1003/train_v1003super.py --no-download --save-dir ckpt_v1003_s0
```

Resolution order: `--data` → `$TENSORMIND_CELEBAHQ` → `~/.cache/tensormind/celebahq256`
→ download. A `--data` that does not check out is an error, never a silent
fallback.

If you bring your own copy, the layout is:

```
/path/to/celebahq256/
  train/       000000.jpg ... 027999.jpg     (28,000)
  validation/  000000.jpg ... 001999.jpg     (2,000)
```

Six digits, and `validation/` restarts at `000000` — it is not a continuation of
the train numbering. The scripts check both counts and refuse to start if either
is wrong.

### Why the download rebuilds the validation order

The public mirror ships its own 28,000/2,000 split, and it is **not** the one the
paper used: measured against the reference copy, the two validation sets agree on
0.5% of positions, i.e. chance. `val_test_split_v1.json` indexes *positionally*
into `sorted(glob('validation/*.jpg'))`, so materialising the mirror's split
as-is would silently score a different 1,000 images and the published numbers
would not reproduce.

`celebahq_val_manifest_v1.npz` therefore holds a re-encode-tolerant fingerprint
of each of the paper's 2,000 validation images, in the paper's order.
`celebahq.py` fingerprints all 30,000 mirror images and rebuilds the exact
evaluation set. Verified end to end: all 2,000 slots match a distinct image at
distance ≤ 19.4, against a typical distance to an unrelated face of 486 — a 25×
margin. If any slot fails to match, the download aborts and writes nothing,
rather than hand you a subtly different benchmark.

Images are written through byte-for-byte from the mirror, with no re-encode, so
there is no second generation of JPEG loss.

## Install

```bash
pip install torch torchvision numpy scikit-image Pillow
pip install lpips torchmetrics torch-fidelity      # for the benchmark and FID
pip install huggingface_hub pyarrow                # only for the dataset download
```

The last line is unnecessary if you pass `--data` and bring your own CelebA-HQ.

## Train

One command per architecture. Each writes checkpoints to `--save-dir`.

```bash
python3 v1003/train_v1003super.py       --save-dir ckpt_v1003_s0
python3 feeder1004/train_feeder1004.py  --save-dir ckpt_feeder1004_s0
python3 unet/train_unetonly_rolling.py  --save-dir ckpt_unet_s0
```

Add `--data /path/to/celebahq256` to any of them to use your own copy instead of
the downloaded one.

**Use a fresh `--save-dir` for every run.** `train_feeder1004.py` says so in its own
docstring: reusing one overwrites the previous run's checkpoints.

The paper trained three seeds per architecture, batch 2, 50 epochs with cosine LR,
and selected the checkpoint by a stopping rule on the selection half of the
validation split.

## Evaluate

```bash
python3 eval_benchmark_v2.py    --checkpoint CKPT.pt --v1003super
python3 eval_benchmark_v2.py    --checkpoint CKPT.pt --feeder1004
python3 eval_benchmark_v2.py    --checkpoint CKPT.pt --unet

python3 eval_fid_2026-08-31.py  --checkpoint CKPT.pt --v1003super
```

Same data resolution as the trainers: `--data /path/to/celebahq256` to use your
own copy, `--val-dir` to point at a bare directory of images, neither to use the
cached or downloaded one.

You must name the architecture — `--unet`, `--v1003super` or `--feeder1004`.
Without one the script exits and tells you so; it cannot infer it from the
checkpoint.

## Automatic downloads

Four things fetch themselves on first use and are then reused. Three are
measuring instruments, not part of the model; the fourth is the dataset.

| what | pulled by | cached in | used for |
|---|---|---|---|
| CelebA-HQ 256 | every script, when no `--data` | `~/.cache/tensormind` | the data itself (~3 GB) |
| VGG16 | all three trainers | `~/.cache/torch` | perceptual loss during training |
| LPIPS | `eval_benchmark_v2.py` | `~/.cache/torch` | the LPIPS column |
| Inception-v3 | `eval_fid_2026-08-31.py` | `~/.cache/torch` | the FID column |

Downloaded once per machine and reused. Training the second architecture does not
re-download VGG16, and the second script does not re-download the dataset.
`--no-download` turns the dataset fetch off; the three networks are not optional.

## The validation split

CelebA-HQ has no unused pool: 28,000 train + 2,000 validation. `val_test_split_v1.json`
divides those 2,000 into a **selection half** (used to choose checkpoints) and a
**reporting half** (used for the published numbers), so the reported metric is not
the one optimised over. Runs finished before 2026-08-15 selected on all 2,000 and
are contaminated on both halves; the paper's nine runs are all later.
