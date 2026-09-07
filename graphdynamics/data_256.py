"""
CelebA-HQ 256x256 dataset with irregular masking for inpainting.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Canonical irregular-mask generator (moved here 2026-08-16).
#
# There were two implementations: this one, which lived in eval_benchmark_v2.py,
# and CelebAHQInpainting._irregular_mask, which drew a fixed 1-4 rectangles and
# 3-8 strokes and then corrected overshoot by restoring random individual pixels.
# That corrector shattered the holes -- measured on 120 validation masks at
# mask_ratio=0.25: median 1,570 connected components, largest hole holding 8% of
# the hidden area, 52% of samples under 10%. So training and validation scored
# local interpolation while the benchmark scored real inpainting.
#
# This version is kept because it is the more general of the two: it draws a
# target first, adds whole elements until the target is reached, and scales
# element size to the target so any coverage from ~2% to ~70% is reachable.
# Training, validation and the benchmark now all call it; the ONLY difference
# between them is the coverage target passed in.
# ─────────────────────────────────────────────────────────────────────────────


def generate_irregular_mask(h, w, rng, target=None):
    """Irregular mask (1=observed, 0=hidden), built incrementally to a TARGET coverage.

    The original version drew a fixed 1-4 rectangles + 3-8 thick strokes with no
    coverage control, which made it impossible to produce masks below ~13% hidden:
    the 0-10% bucket was always empty and 10-20% could only be filled with ~20
    outliers at ~18% mean coverage. Here a target is drawn first and elements are
    added one at a time until it is reached, so every bucket is reachable and the
    element vocabulary (rectangles + brush strokes) is unchanged.
    """
    if target is None:
        target = rng.uniform(0.02, 0.70)
    mask = np.ones((h, w), dtype=np.float32)

    def cov():
        return 1.0 - mask.mean()

    def add_rect(scale):
        rect_h = max(4, int(rng.integers(h // 16, max(h // 16 + 1, int(h * scale)))))
        rect_w = max(4, int(rng.integers(w // 16, max(w // 16 + 1, int(w * scale)))))
        y0 = rng.integers(0, max(1, h - rect_h))
        x0 = rng.integers(0, max(1, w - rect_w))
        mask[y0:y0 + rect_h, x0:x0 + rect_w] = 0.0

    def add_stroke(thickness, n_steps):
        y, x = rng.integers(0, h), rng.integers(0, w)
        yy, xx = np.ogrid[-thickness:thickness + 1, -thickness:thickness + 1]
        circle = yy ** 2 + xx ** 2 <= thickness ** 2
        for _ in range(n_steps):
            y_start, x_start = max(0, y - thickness), max(0, x - thickness)
            y_end, x_end = min(h, y + thickness + 1), min(w, x + thickness + 1)
            cy_start, cx_start = max(0, thickness - y), max(0, thickness - x)
            cy_end, cx_end = cy_start + (y_end - y_start), cx_start + (x_end - x_start)
            mask[y_start:y_end, x_start:x_end][
                circle[cy_start:cy_end, cx_start:cx_end]] = 0.0
            angle = rng.uniform(0, 2 * np.pi)
            step = rng.integers(5, 20)
            y = int(np.clip(y + step * np.sin(angle), 0, h - 1))
            x = int(np.clip(x + step * np.cos(angle), 0, w - 1))

    # scale element size to the target so light masks are reachable
    scale = float(np.clip(0.15 + 0.5 * target, 0.10, 0.50))
    guard = 0
    while cov() < target and guard < 60:
        guard += 1
        if rng.random() < 0.4:
            add_rect(scale)
        else:
            thickness = int(rng.integers(4, max(5, int(6 + 40 * target))))
            n_steps = int(rng.integers(5, max(6, int(10 + 90 * target))))
            add_stroke(thickness, n_steps)
    return mask


class CelebAHQInpainting(Dataset):
    """
    Loads CelebA-HQ 256x256 images and applies irregular masks.

    Returns dict with:
        x:     (3, 256, 256) ground truth in [0, 1]
        x_obs: (3, 256, 256) observed image (masked region = 0)
        m:     (1, 256, 256) binary mask (1 = observed, 0 = hidden)
    """

    def __init__(self, root, split='train', mask_ratio=0.25, irregular=False):
        super().__init__()
        self.root = root
        self.split = split
        self.mask_ratio = mask_ratio
        self.irregular = irregular

        # Look for split subdirectories first, then fall back to flat
        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            files = sorted(glob.glob(os.path.join(split_dir, '*.jpg')))
            if not files:
                files = sorted(glob.glob(os.path.join(split_dir, '*.png')))
        else:
            files = sorted(glob.glob(os.path.join(root, '*.jpg')))
            if not files:
                files = sorted(glob.glob(os.path.join(root, '*.png')))
            # 90/10 train/val split if no subdirs
            n = len(files)
            split_idx = int(0.9 * n)
            if split == 'train':
                files = files[:split_idx]
            else:
                files = files[split_idx:]

        if not files:
            raise FileNotFoundError(f"No images found in {root} for split={split}")

        self.files = files

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ])

        print(f"CelebA-HQ {split}: {len(self.files)} images, mask_ratio={mask_ratio}")

    def __len__(self):
        return len(self.files)

    # ---- REMOVED 2026-08-16: _random_mask ----
    # A single axis-aligned rectangle, reachable only when irregular=False.
    # Every training and evaluation script constructs this dataset with
    # irregular=True, so it has never run in any experiment in this line.
    # was: def _random_mask(self, h, w, rng): ... single rect of area
    #      h*w*mask_ratio with aspect ~ U(0.5, 2.0)

    def _irregular_mask(self, h, w, rng):
        """Draw a coverage target from the mask_ratio band, then defer to the
        canonical generator above. Replaced 2026-08-16; the previous body (fixed
        geometry + per-pixel overshoot correction, and its 2026-08-16 incremental
        rewrite) is superseded by the shared implementation."""
        lo = max(0.09, self.mask_ratio - 0.16)
        hi = min(0.8, self.mask_ratio + 0.24)
        target_ratio = rng.uniform(lo, hi)
        mask = generate_irregular_mask(h, w, rng, target=target_ratio)
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        x = self.transform(img)  # (3, 256, 256)

        if self.split == 'train':
            # ---- CHANGED 2026-08-16: SeedSequence instead of an integer sum ----
            # was: default_rng(idx + int(torch.randint(0, 100000, (1,)).item()))
            # Summing collapses the seed space: idx in [0,28000) plus an offset in
            # [0,100000) reaches only ~128,000 distinct values, and (idx, offset)
            # pairs collide heavily -- over a run of 14,000 steps x 2 x 50 epochs
            # = 1.4M draws every mask/dropout/distortion realisation recurs about
            # 11 times. It also correlates the mask with the image index, since
            # nearby idx with the same offset give nearby seeds. A SeedSequence
            # spawn keys on the two numbers independently, so the space is the
            # full 2^128 and idx and offset are decorrelated. The per-epoch offset
            # still comes from the torch stream, so runs stay reproducible.
            _off = int(torch.randint(0, 2**31 - 1, (1,)).item())
            rng = np.random.default_rng(np.random.SeedSequence([idx, _off]))
        else:
            rng = np.random.default_rng(idx)  # deterministic for validation
        # `irregular` is retained as a constructor argument for compatibility;
        # only the irregular generator exists now (see the removal note above).
        if not self.irregular:
            raise ValueError("irregular=False is no longer supported; _random_mask "
                             "was removed 2026-08-16 as unreachable dead code.")
        # ---- ADDED 2026-08-16 (a): photometric augmentation acts on the CLEAN
        #      image, before masking ----
        # Brightness and contrast used to be applied to `x_obs` only, with the
        # target `x` left clean (see the note in _random_distort). That is not
        # augmentation, it is an unidentifiable corruption: given only x_obs the
        # network cannot recover `delta` or `factor`, so the best it can do on
        # the observed term is regress toward the population mean. Measured over
        # 200 real train samples: 52% of samples carried a photometric change and
        # mean |x_obs - x| over observed pixels was 0.0944, a floor no network
        # can beat. Worse, it reached the PRIMARY metric and not just the obs
        # term -- on the 26% of samples with a brightness shift the whole visible
        # context was offset by up to +/-0.15 while the hidden-region target was
        # not, training the network to distrust absolute brightness and then
        # scoring it where absolute brightness is informative.
        #
        # Applying the same transform to `x` and `x_obs` together restores
        # identifiability: the target is the augmented image, which is exactly
        # what the input is a masked view of, so the PHOTOMETRIC part of the
        # floor goes to zero. (Corrected 2026-08-16: an earlier draft of this
        # note said "a perfect network scores 0 again", which is false. Noise,
        # blur and dropout still separate x_obs from x -- measured 0.0630 over
        # observed pixels, of which 0.0135 is noise+blur and the rest is the
        # blanked pixels. That floor is intended: unlike a global brightness
        # shift, each of those is recoverable from the image itself, which is
        # what makes the observed term a real denoising objective rather than an
        # unanswerable one. train_obs will therefore sit above val_obs, and that
        # is not a bug.)
        # This is the standard use of a photometric transform, and it keeps the
        # regularisation the transform was there to provide.
        #
        # The clean split now is: photometric on `x` (augmentation, here);
        # noise / blur / pixel-dropout on `x_obs` only (corruption, below).
        if self.split == 'train':
            x = self._photometric(x, rng)

        m = self._irregular_mask(x.shape[1], x.shape[2], rng)

        # Training: input dropout of a random fraction of the OBSERVED pixels.
        # (The old fixed 0.1 * H * W and its "14.3% of observed" gloss are both
        # superseded -- see the note below.)
        # ---- CHANGED 2026-08-16: dropout corrupts the INPUT, not the MASK ----
        # It used to set m = 0 at the dropped pixels, which put ~6,553 isolated
        # single pixels into the hidden region of every training mask. Measured:
        # connected components median 8 (mask alone) -> 4,501 after dropout, and
        # 27.7% of all hidden pixels were isolated singles. So even after the
        # generator was fixed, training still scored ~28% trivial
        # interpolate-from-four-neighbours while validation scored none of it,
        # and hidden fraction was 0.40 in training vs 0.31 in validation.
        # The augmentation is kept -- the pixels are still blanked in x_obs, so
        # the network sees a corrupted input -- but m is left alone, so the miss
        # term scores real holes only and matches validation. Reconstructing the
        # blanked pixels is supervised by the observed-region term, which since
        # the 2026-08-16 obs-on-raw change is a real denoising objective.
        # was: m[0, obs_y[drop_idx], obs_x[drop_idx]] = 0.0
        # ---- CHANGED 2026-08-16 (b): dropout is a FRACTION OF OBSERVED PIXELS,
        #      drawn per sample from U(0, 0.25) ----
        # Two defects in the version above.
        #
        # (i) `n_drop = 0.1 * H * W` is a fraction of the IMAGE, so the fraction
        #     of the *context* it destroys depends on the mask. At 9% hidden it
        #     blanks 11.0% of the observed pixels; at 53% hidden it blanks 21.3%.
        #     Corruption strength was anti-correlated with how much context the
        #     sample had left -- the hardest samples were also corrupted hardest,
        #     which is the opposite of a controlled variable. Measured mean was
        #     17.7% of observed, not the 14.3% the old comment claimed (mean
        #     hidden fraction is 0.317, not 0.25, and the photometric clamp
        #     zeroed further observed pixels -- that second cause is gone now
        #     that photometric acts on x, see (a) above).
        #
        # (ii) It fired on EVERY training sample. The network therefore never
        #      saw an uncorrupted context in training, while every benchmark and
        #      validation sample has one (`eval_benchmark_v2.py` does
        #      `x_obs = gt * mask_t`, no dropout). The evaluation condition sat
        #      strictly outside the training distribution.
        #
        # Drawing the rate from U(0, 0.25) per sample fixes both: it is defined
        # against the observed count so it is mask-invariant, and 0 is inside the
        # support, so the eval condition is covered rather than extrapolated. The
        # expected rate is close to the old effective one. (Two numbers appear
        # for the old rate in this block: 17.7% is what was measured end-to-end,
        # 14.6% is the dropout mechanism alone -- the difference is observed
        # pixels that the photometric clamp drove to exactly zero, which the
        # measurement could not tell apart from dropped ones. That second cause
        # no longer exists.) The expected rate here, 12.5% before the clean
        # branch below and 10% after it,
        # so regularisation strength is roughly preserved rather than silently
        # retuned.
        # was: n_drop = int(0.1 * x.shape[1] * x.shape[2])
        input_dropout = None
        if self.split == 'train':
            obs_y, obs_x = torch.where(m[0] == 1)
            n_obs = len(obs_y)
            # ---- CHANGED 2026-08-16 (b2): an explicit clean-context branch ----
            # The uniform draw alone did NOT achieve what the note above claims.
            # With n_obs ~ 45,000, P(int(U(0,0.25) * n_obs) == 0) is about
            # 2.2e-5, i.e. ~1.4 samples in a 14,000-step epoch. Zero was a limit
            # point of the support, not a covered condition, so the network
            # still essentially never saw an uncorrupted context while every
            # benchmark and validation sample has one -- which was the whole of
            # defect (ii). One sample in five is now clean outright. Expected
            # dropout becomes 0.8 * 12.5% = 10% of observed.
            #
            # NOTE 2026-08-16: when this branch was first added its whole block
            # was pasted at 8-space indent, i.e. OUTSIDE the `if self.split ==
            # 'train'` guard above. Training was unaffected (n_obs is assigned
            # two lines up), so a train-split measurement looked correct -- but
            # every validation sample hit `n_obs` unbound and both runs died at
            # the first end-of-epoch validation, 20 and 33 minutes in. Anything
            # touching this block must be tested on BOTH splits.
            # was: n_drop = int(rng.uniform(0.0, 0.25) * n_obs)
            if rng.random() < 0.2:
                n_drop = 0
            else:
                n_drop = int(rng.uniform(0.0, 0.25) * n_obs)
            if n_drop > 0 and n_obs > n_drop:
                drop_idx = rng.choice(n_obs, size=n_drop, replace=False)
                input_dropout = torch.ones_like(m)
                input_dropout[0, obs_y[drop_idx], obs_x[drop_idx]] = 0.0

        x_obs = x * m

        # ---- REORDERED 2026-08-16: distort first, then drop pixels ----
        # The dropout used to run first and the distortions after, which partly
        # undid it: the brightness branch leaves a dropped pixel at `delta`, the
        # contrast branch at `mean*(1-factor)`, and the blur smears neighbours
        # back into it. Measured 8.8% of observed pixels still exactly black
        # against the 14.3% the dropout nominally blanks. Dropping last means the
        # blanked pixels are genuinely blank, which is what the comment above
        # claims and what the obs term is supposed to be denoising.
        # was: x_obs = x*m ; x_obs *= input_dropout ; x_obs = _random_distort(...)
        if self.split == 'train':
            x_obs = self._random_distort(x_obs, m, rng)
        if input_dropout is not None:
            x_obs = x_obs * input_dropout

        return {'x': x, 'x_obs': x_obs, 'm': m}

    def _photometric(self, x, rng):
        """Brightness / contrast on the whole clean image. ADDED 2026-08-16.

        Applied before masking, so `x` (the target) and `x_obs` (the input) carry
        the same transform and the mapping stays identifiable. Same draw
        probabilities and same ranges as the branches this replaces, so the
        strength of the augmentation is unchanged -- only its target is.
        """
        if rng.random() < 0.3:
            delta = rng.uniform(-0.15, 0.15)
            x = (x + delta).clamp(0, 1)

        if rng.random() < 0.3:
            factor = rng.uniform(0.7, 1.3)
            # Full-image mean. The old version took the mean over observed
            # pixels only, which made a global photometric transform depend on
            # the mask -- two different masks on one image gave two different
            # contrast pivots.
            mean = x.mean()
            x = ((x - mean) * factor + mean).clamp(0, 1)

        return x

    def _random_distort(self, x_obs, m, rng):
        """Corruption of the observed pixels only. The target stays clean.

        Only sensor-style corruptions belong here -- ones the network can undo
        from the image itself. Brightness and contrast moved to _photometric on
        2026-08-16 (see the note in __getitem__); they were unidentifiable here.
        """
        if rng.random() < 0.3:
            sigma = rng.uniform(0.02, 0.08)
            noise = torch.from_numpy(rng.normal(0, sigma, x_obs.shape).astype(np.float32))
            x_obs = (x_obs + noise * m).clamp(0, 1)

        # ---- MOVED 2026-08-16 -> _photometric(), applied to x and x_obs together
        # was: if rng.random() < 0.3:
        # was:     delta = rng.uniform(-0.15, 0.15)
        # was:     x_obs = (x_obs + delta * m).clamp(0, 1)
        #
        # was: if rng.random() < 0.3:
        # was:     factor = rng.uniform(0.7, 1.3)
        # was:     mean = (x_obs * m).sum() / max(m.sum() * 3, 1)
        # was:     x_obs = ((x_obs - mean) * factor + mean) * m
        # was:     x_obs = x_obs.clamp(0, 1)

        # ---- REMOVED 2026-08-16: whole-channel drop ----
        # This zeroed an entire colour channel on 10% of training samples.
        # It was free while the observed-region loss term had structurally zero
        # gradient, but that term now supervises `raw` against the CLEAN target,
        # so it asks the network to reconstruct a deleted colour channel from
        # the other two -- information-theoretically impossible. It put an
        # irreducible floor under train_obs (measured: 12% of samples affected,
        # mean |x_obs - x| over the observed region 0.085) and made train_obs
        # and val_obs measure different things. The recoverable corruptions
        # (noise and blur) are kept -- those are a genuine denoising objective.
        # (Brightness and contrast were listed here as "recoverable" too. They
        # are not, and on 2026-08-16 they were moved to _photometric where they
        # act on the target as well; this note was wrong about two of the four.)
        # was: if rng.random() < 0.1:
        # was:     ch = rng.integers(0, 3)
        # was:     x_obs[ch] = 0.0

        if rng.random() < 0.2:
            k = rng.choice([3, 5])
            sigma_b = rng.uniform(0.5, 1.5)
            from torch.nn.functional import conv2d
            ax = torch.arange(k, dtype=torch.float32) - k // 2
            kernel = torch.exp(-ax**2 / (2 * sigma_b**2))
            kernel = (kernel.unsqueeze(0) * kernel.unsqueeze(1))
            kernel = kernel / kernel.sum()
            kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, k, k)
            pad = k // 2
            # ---- CHANGED 2026-08-16: normalise by the blurred mask ----
            # was: blurred = conv2d(x_obs, kernel, ...); x_obs = blurred * m
            # x_obs is zero inside the hole, so a plain convolution averaged those
            # zeros into observed pixels within k//2 of the boundary, darkening a
            # ring around every hole. The obs term then asked the network to undo
            # that darkening -- impossible, since the network cannot know how much
            # hole was in each kernel footprint. Dividing by the identically
            # blurred mask renormalises each output by the observed weight that
            # actually contributed, so the blur stays inside the observed region.
            mm = m.expand_as(x_obs)
            blurred = conv2d((x_obs * mm).unsqueeze(0), kernel, padding=pad, groups=3)[0]
            wsum = conv2d(mm.unsqueeze(0), kernel, padding=pad, groups=3)[0]
            x_obs = (blurred / wsum.clamp(min=1e-6)) * m

        return x_obs
