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
data_256.py                                   dataset + irregular mask generation
eval_benchmark_v2.py                          PSNR / SSIM / LPIPS / L1 by mask coverage
eval_fid_2026-08-31.py                        FID
```

The graph and the split are the defaults for `--graph` and `--split-file`, so you
do not need to pass either. Both resolve relative to the script, so any working
directory works.

## What you must supply

**CelebA-HQ 256x256**, as `--data`. Nothing downloads it for you. Layout:

```
/path/to/celebahq256/
  train/       00000.jpg ...
  validation/  28000.jpg ...
```

## Install

```bash
pip install torch torchvision numpy scikit-image Pillow
pip install lpips torchmetrics torch-fidelity      # for the benchmark and FID
```

## Train

One command per architecture. Each writes checkpoints to `--save-dir`.

```bash
python3 v1003/train_v1003super.py       --data /path/to/celebahq256 --save-dir ckpt_v1003_s0
python3 feeder1004/train_feeder1004.py  --data /path/to/celebahq256 --save-dir ckpt_feeder1004_s0
python3 unet/train_unetonly_rolling.py  --data /path/to/celebahq256 --save-dir ckpt_unet_s0
```

**Use a fresh `--save-dir` for every run.** `train_feeder1004.py` says so in its own
docstring: reusing one overwrites the previous run's checkpoints.

The paper trained three seeds per architecture, batch 2, 50 epochs with cosine LR,
and selected the checkpoint by a stopping rule on the selection half of the
validation split.

## Evaluate

```bash
python3 eval_benchmark_v2.py --checkpoint CKPT.pt --v1003super  --data /path/to/celebahq256
python3 eval_benchmark_v2.py --checkpoint CKPT.pt --feeder1004  --data /path/to/celebahq256
python3 eval_benchmark_v2.py --checkpoint CKPT.pt --unet        --data /path/to/celebahq256

python3 eval_fid_2026-08-31.py --checkpoint CKPT.pt --v1003super
```

You must name the architecture — `--unet`, `--v1003super` or `--feeder1004`.
Without one the script exits and tells you so; it cannot infer it from the
checkpoint.

## Automatic downloads

Three pretrained networks fetch themselves on first use and cache in
`~/.cache/torch`. They are measuring instruments, not part of the model.

| what | pulled by | used for |
|---|---|---|
| VGG16 | all three trainers | perceptual loss during training |
| LPIPS | `eval_benchmark_v2.py` | the LPIPS column |
| Inception-v3 | `eval_fid_2026-08-31.py` | the FID column |

Downloaded once per machine and reused. Training the second architecture does not
re-download VGG16.

## The validation split

CelebA-HQ has no unused pool: 28,000 train + 2,000 validation. `val_test_split_v1.json`
divides those 2,000 into a **selection half** (used to choose checkpoints) and a
**reporting half** (used for the published numbers), so the reported metric is not
the one optimised over. Runs finished before 2026-08-15 selected on all 2,000 and
are contaminated on both halves; the paper's nine runs are all later.
