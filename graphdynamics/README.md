# graphdynamics

The inpainting models, their training scripts, and the two evaluation scripts.

| paper row | model | trainer | status |
|---|---|---|---|
| Graph${+}$attn. | `v1003/model_v1003super.py` | `v1003/train_v1003super.py` | **reported** |
| U-Net | `unet/model_unetonly.py` | `unet/train_unetonly_rolling.py` | **baseline** |
| Graph (no attn.) | `feeder1004/feeder1004.py` | `feeder1004/train_feeder1004.py` | ablation; not in the main article |

`feeder1004` is `v1003super` with the supernode attention removed — the ablation.
Its header records the md5 of the file it was copied from and the three lines
that differ. The main article reports only the attention arm against the U-Net
and does not mention this ablation. The supplementary notes mention it once, for
the auxiliary 16,807-node substrate ("a single run with the attention blocks
removed performed about the same"); see *Results* below.

## What ships here

```
graph_degmatch_parent_supernode.npz          16,807 nodes / 337,980 edges, 1,015 supernodes  <- the DEFAULT here
graph_brain_mild_v146_feeder_supernode.npz   16,807 nodes / 344,483 edges, 1,015 supernodes  (superseded)
val_test_split_v1.json                       which 1,000 validation images are the reporting half
celebahq.py                                   dataset resolution + optional download
celebahq_val_manifest_v1.npz                  fingerprints of the 2,000 validation images, in order
data_256.py                                   dataset + irregular mask generation
eval_benchmark_v2.py                          PSNR / SSIM / LPIPS / L1 by mask coverage
eval_fid_2026-08-31.py                        FID
```

## The 8,000-node substrate — build it first

`graph_8k_parent_supernode.npz` is **not shipped**: like every other graph in
this repository it is regenerable, so `.gitignore` keeps it out. The two graphs
listed above are the exceptions. Build the 8k one before using it:

```bash
cd ../graphnets/construction/bilateral
python3 export_graphs.py                  # ~3 min, deterministic, needs ~3 GB
cp graphs/cnew_parent_supernode.npz ../../../graphdynamics/graph_8k_parent_supernode.npz
cd ../../../graphdynamics
```

`export_graphs.py` runs `construct.py`'s own `grow()` and `coarsen()` and writes
the four files the repository reads — the parent, the parent with its supernode
assignment, the 1,015-node quotient, and the quotient's bundle weights. It is
deterministic from the hard-coded seed, so every run gives the same graph;
`python3 export_graphs.py --check` verifies a rebuild against files already
present, comparing array contents rather than file bytes (a `.npz` is a zip and
timestamps every entry, so checksums of identical arrays never match).

The same four files are what `bilateral/stats.py`, `bilateral/bundles/bundles.py`
and `bilateral/make_figures.py` read, so build them before running those too.

Then train or evaluate on that substrate by naming it explicitly — nothing in
this directory defaults to it:

```bash
python3 v1003/train_v1003super.py      --graph graph_8k_parent_supernode.npz --save-dir ckpt_8k_v1003_s0 --seed 0 --from-scratch
python3 feeder1004/train_feeder1004.py --graph graph_8k_parent_supernode.npz --save-dir ckpt_8k_feeder_s0 --seed 0 --from-scratch
python3 eval_benchmark_v2.py --checkpoint CKPT.pt --v1003super --graph graph_8k_parent_supernode.npz --mask-dist training
```

The U-Net baseline takes no graph at all, so it is the same network on either
substrate and needs nothing extra.

**Which substrate the paper reports.** The main article's three-seed inpainting
comparison is on this 8,000-node substrate; the `graph_degmatch_parent_supernode`
results are the auxiliary experiment described in the supplementary notes. The
defaults in this directory stay on `graph_degmatch_parent_supernode.npz` for
continuity with the earlier runs, so reproducing the paper's numbers means
passing `--graph graph_8k_parent_supernode.npz` as above.

Both the graph and the split are defaults, so `--graph` and `--split-file` can be
omitted. The commands below pass `--graph` anyway, to be explicit about which
substrate each number belongs to.

`graph_degmatch_parent_supernode.npz` is the final parent built by
`graphnets/construction/finalgraph/` — a compressed copy of its
`parent_degmatch_masks.npz`, arrays identical, supernode map included.
Rebuild it there if you want to check.

It is the DEFAULT, not the substrate the article reports. It was marked
"REPORTED" here until 2026-09-25, which predated the article being reorganised
around the 8,000-node construction; the 16,807-node results are the auxiliary
experiment the supplementary notes mark exploratory. The default stays here for
continuity with the earlier runs — see the section above for how to build and use
the 8k substrate.

`graph_brain_mild_v146_feeder_supernode.npz` is the feeder lift the earlier runs
used. It is kept only so those numbers stay reproducible; pass it explicitly to
get them. Nothing defaults to it any more.

Both resolve relative to the script, so any working directory works.

## The data

**CelebA-HQ 256x256.** You do not have to supply it: with no `--data`, every
script here fetches a pinned copy on first use and reuses it afterwards.

```bash
# nothing to do -- downloads ~3 GB to ~/.cache/tensormind/celebahq256, once
python3 v1003/train_v1003super.py --graph graph_degmatch_parent_supernode.npz --save-dir ckpt_v1003_s0 --from-scratch

# or point at your own copy
python3 v1003/train_v1003super.py --graph graph_degmatch_parent_supernode.npz \
    --data /path/to/celebahq256 --save-dir ckpt_v1003_s0 --from-scratch

# or set it once for every script
export TENSORMIND_CELEBAHQ=/path/to/celebahq256

# or refuse to download and fail loudly instead
python3 v1003/train_v1003super.py --graph graph_degmatch_parent_supernode.npz --no-download --save-dir ckpt_v1003_s0 --from-scratch
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
python3 v1003/train_v1003super.py      --graph graph_degmatch_parent_supernode.npz --save-dir ckpt_v1003_s0 --from-scratch
python3 feeder1004/train_feeder1004.py --graph graph_degmatch_parent_supernode.npz --save-dir ckpt_feeder1004_s0 --from-scratch
python3 unet/train_unetonly_rolling.py --save-dir ckpt_unet_s0   # no graph
```

Add `--data /path/to/celebahq256` to any of them to use your own copy instead of
the downloaded one.

**Use a fresh `--save-dir` for every run.** `train_feeder1004.py` says so in its own
docstring: reusing one overwrites the previous run's checkpoints.

Seeds are set with `--seed`; use a matching `--save-dir`:

```bash
python3 v1003/train_v1003super.py --graph graph_degmatch_parent_supernode.npz \
    --seed 1 --save-dir ckpt_v1003_s1 --from-scratch
```

Batch 2, 50 epochs, cosine LR. There is no early stopping: every run trains the full 50 epochs. The reported
checkpoint is chosen by a pre-declared rule on the selection half of the
validation split — **the first epoch after epoch 40 at which three consecutive
epochs report the same validation loss to four decimal places**; the first of
the three is kept. This is the wording that reproduces every reported epoch from
the surviving training logs (8k graph seed 2 → 45; U-Net seeds 0, 1, 2 → 47, 46,
46). The main article words the rule as "the last epoch to improve the
validation loss at four decimal places before three consecutive epochs without
improvement", which selects the same epochs except U-Net seed 0, where only two
epochs follow its last improvement (epoch 47) so a strict reading never fires.

## Results — the 8,000-node substrate (the main article)

Three seeds per family, scored by `eval_benchmark_v2.py --mask-dist training` on
the reporting half of the validation split: 1,000 images × 6 masks = 6,000
samples per run, masks drawn from the training band `U(0.09, 0.49)`. Values are
means over all 6,000 samples. The wider `U(0.02, 0.70)` band is not used for the
main article's comparison.

| model | seed | checkpoint | L1 % | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| U-Net | 0 / 1 / 2 | ep47 / 46 / 46 | 1.462 / 1.478 / 1.471 | 28.42 / 28.34 / 28.38 | 0.8986 / 0.8979 / 0.8980 | 0.1076 / 0.1084 / 0.1086 |
| U-Net | mean | | 1.470 | 28.38 | 0.8982 | 0.1082 |
| Graph+attn. | 0 / 1 / 2 | ep46 / 45 / 45 | 1.362 / 1.370 / 1.364 | 28.92 / 28.88 / 28.91 | 0.9073 / 0.9069 / 0.9072 | 0.0911 / 0.0913 / 0.0911 |
| Graph+attn. | mean | | **1.365** | **28.90** | **0.9071** | **0.0912** |

Pooled FID: 9.84 ± 0.07 (graph) against 11.53 ± 0.19 (U-Net), from
`eval_fid_2026-08-31.py`. The per-sample JSONs these were re-aggregated from are
listed in the main article's data index; they are not shipped here.

## Results — the 16,807-node substrate (auxiliary, supplementary notes)

All on the final parent, seed 0, checkpoint chosen by the rule above, scored by
`eval_benchmark_v2.py` on the reporting half of the validation split. Mean over
the six coverage buckets. On these runs the rule selects epoch 45 (with
attention) and epoch 47 (without).

**Benchmark mask distribution** — `U(0.02, 0.70)`:

| | L1 % | PSNR dB | SSIM | LPIPS |
|---|---|---|---|---|
| Graph${+}$attn. (ep45) | 1.4506 | 29.826 | 0.90403 | 0.09329 |
| Graph, no attn. (ep47) | 1.4507 | 29.843 | 0.90415 | 0.09332 |

**Training mask distribution** — `--mask-dist training`, `U(0.09, 0.49)`:

| | L1 % | PSNR dB | SSIM | LPIPS |
|---|---|---|---|---|
| Graph${+}$attn. (ep45) | 1.4099 | 29.352 | 0.90553 | 0.09179 |
| Graph, no attn. (ep47) | 1.4088 | 29.389 | 0.90565 | 0.09174 |

**The attention makes no measurable difference.** L1 differs by −0.0000 on
benchmark masks and +0.0011 on training masks — both far inside the seed spread
(sd 0.0030 across baseline seeds), and the sign is not stable between the two.
Final validation loss: 0.043542 with attention, 0.043599 without, a gap of
5.7e-05.

**On reading the two distributions.** The aggregate moves a lot between them
(L1 1.4506 → 1.4099) but that is a change in the sample mix, not in model
behaviour. The training band's floor at 0.09 starves the 0-10% bucket
(652 → 64 samples) and its target ceiling at 0.49 starves 50-60%
(827 → 240, reached only by overshoot). The four middle buckets, which both
bands populate properly, agree to within 0.01. On this auxiliary substrate the
supplementary notes report the benchmark band as primary and the training band as
a robustness check. The main article's 8k comparison above uses the training
band only.

**Seeds.** The attention arm on this substrate has three seeds
(`ckpt_v1003super_degmatch`, `_s1`, `_s2`). The no-attention arm has one run
(`ckpt_feeder1004_degmatch`), as the supplementary notes say. There is no
no-attention run on the 8k substrate. (Until 2026-10-06 this paragraph said all
three seeds of both arms had been trained on both substrates; the checkpoint
directories do not support that.) Seed spread on this benchmark is ~0.003 in L1,
so a single seed resolves nothing below ~0.006 — which is why the attention claim
rests on the direction of the difference being unstable rather than on its size.


## Evaluate

```bash
python3 eval_benchmark_v2.py    --checkpoint CKPT.pt --v1003super --graph graph_degmatch_parent_supernode.npz
python3 eval_benchmark_v2.py    --checkpoint CKPT.pt --feeder1004 --graph graph_degmatch_parent_supernode.npz
python3 eval_benchmark_v2.py    --checkpoint CKPT.pt --unet

python3 eval_fid_2026-08-31.py  --checkpoint CKPT.pt --v1003super --graph graph_degmatch_parent_supernode.npz
```

**Mask distribution.** The main article's numbers use `--mask-dist training`.
For the auxiliary substrate, evaluate on both. Training draws its coverage target from
`U(0.09, 0.49)`; the benchmark's default draws from `U(0.02, 0.70)`. They use the
same generator with different inputs, and 11.8% of scored samples fall outside
the training range — all below its floor, all in the 0-20% band. Add
`--mask-dist training` for the second pass:

```bash
python3 eval_benchmark_v2.py --checkpoint CKPT.pt --v1003super --graph graph_degmatch_parent_supernode.npz --mask-dist training
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
