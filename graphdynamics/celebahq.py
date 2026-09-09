#!/usr/bin/env python3
"""CelebA-HQ 256x256 resolution, with optional automatic download.

Every entry point in this package (the three trainers, the benchmark, the FID
script) takes an optional data path.  If one is given it is used and nothing is
downloaded.  If none is given, this module resolves a path in the following
order, downloading only as a last resort:

    1. the explicit argument (--data, or --val-dir for the eval scripts)
    2. $TENSORMIND_CELEBAHQ
    3. ~/.cache/tensormind/celebahq256, if already materialised
    4. download to (3)

The download is a pinned revision of a public mirror, so two people running it
a year apart get the same bytes.


WHY THE VALIDATION ORDER IS REBUILT RATHER THAN INHERITED
---------------------------------------------------------
The mirror ships its own 28,000 / 2,000 train/validation partition, and it is
NOT the one the paper used: measured against the reference copy, the two
validation sets agree on 0.5% of entries, i.e. chance.  `val_test_split_v1.json`
indexes positionally into `sorted(glob('validation/*.jpg'))`, so materialising
the mirror's split as-is would silently score a different 1,000 images and the
published numbers would not reproduce.

`celebahq_val_manifest_v1.npz` therefore holds a re-encode-tolerant
fingerprint of each of the paper's 2,000 validation images, in the paper's
order.  The downloader fingerprints all 30,000 mirror images, assigns each
manifest slot its nearest match, writes those as validation/000000.jpg ...
001999.jpg, and writes the remaining 28,000 as train/.  Training order is
irrelevant (the loader shuffles every epoch); validation order is not.

Measured on the reference copy against this mirror: every one of the 2,000
slots matches at distance <= 19.4 with a distinct target, while the typical
distance to the *second* nearest image is 486 -- a 25x margin.  Two of the
2,000 match an image the mirror files under train/ rather than validation/;
both are at distance ~9, i.e. genuine near-duplicate pairs inside CelebA-HQ,
so either member reconstructs the same picture.

If the manifest is missing, or any slot fails to match, the downloader refuses
to guess: it writes nothing and tells you to pass --data.  A silently different
evaluation set is worse than no data at all.


FINGERPRINT STABILITY
---------------------
The resample filter is pinned to BILINEAR.  Pillow changed its default filter,
and the same image fingerprinted under the two filters gives distances of ~72
rather than ~9 -- still separable here, but not a margin to leave to chance.
Do not "simplify" the resize call.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys

__all__ = ["resolve", "resolve_val_dir", "is_materialised", "DEFAULT_CACHE"]

# Pinned mirror: train 28,000 + validation 2,000, 256x256, 30,000 total.
REPO = "korexyz/celeba-hq-256x256"
REVISION = "42078a1fb27b6079eaab8cf453efd3d79a8b7bda"
PARQUETS = [f"data/train-0000{i}-of-00006.parquet" for i in range(6)] + [
    "data/validation-00000-of-00001.parquet"
]

N_TRAIN = 28000
N_VAL = 2000

# Reject a match beyond this.  Observed worst true match 19.4; typical distance
# to an unrelated face 486.  40 sits an order of magnitude inside the gap.
MATCH_MAX = 40.0

DEFAULT_CACHE = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
    "tensormind", "celebahq256")

_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "celebahq_val_manifest_v1.npz")


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

def fingerprint(img):
    """16x16 greyscale, BILINEAR, as uint8 -- see FINGERPRINT STABILITY above."""
    import numpy as np
    from PIL import Image
    return np.asarray(img.convert("L").resize((16, 16), Image.BILINEAR),
                      dtype=np.uint8).ravel()


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def _n_images(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))


def is_materialised(root):
    """True when `root` holds train/ and validation/ at the expected counts."""
    if not root or not os.path.isdir(root):
        return False
    return (_n_images(os.path.join(root, "train")) == N_TRAIN
            and _n_images(os.path.join(root, "validation")) == N_VAL)


def _describe(root):
    return (f"{root}\n"
            f"      train/      {_n_images(os.path.join(root, 'train')):,} images "
            f"(expected {N_TRAIN:,})\n"
            f"      validation/ {_n_images(os.path.join(root, 'validation')):,} images "
            f"(expected {N_VAL:,})")


def resolve(root=None, download=True, quiet=False):
    """Return a directory holding train/ and validation/, downloading if needed.

    `root` is whatever the caller's --data was, or None.  An explicit root that
    does not check out is an error, never a silent fallback: if you pointed the
    script at a directory, you meant that directory.
    """
    if root:
        root = os.path.expanduser(root)
        if is_materialised(root):
            return root
        raise SystemExit(
            f"--data does not look like CelebA-HQ 256:\n      {_describe(root)}\n\n"
            f"    Expected {root}/train/ and {root}/validation/.\n"
            f"    Omit --data to download a pinned copy to {DEFAULT_CACHE}.")

    env = os.environ.get("TENSORMIND_CELEBAHQ")
    if env:
        env = os.path.expanduser(env)
        if is_materialised(env):
            if not quiet:
                print(f"CelebA-HQ: $TENSORMIND_CELEBAHQ -> {env}")
            return env
        raise SystemExit(
            f"$TENSORMIND_CELEBAHQ is set but does not look like CelebA-HQ 256:\n"
            f"      {_describe(env)}")

    if is_materialised(DEFAULT_CACHE):
        if not quiet:
            print(f"CelebA-HQ: cached at {DEFAULT_CACHE}")
        return DEFAULT_CACHE

    if not download:
        raise SystemExit(
            f"No CelebA-HQ found and --no-download was given.\n"
            f"    Pass --data /path/to/celebahq256, or drop --no-download to fetch\n"
            f"    a pinned copy (~3 GB) to {DEFAULT_CACHE}.")

    return _download(DEFAULT_CACHE, quiet=quiet)


def resolve_val_dir(val_dir=None, root=None, download=True, quiet=False):
    """As `resolve`, for the eval scripts, which want the validation directory.

    An explicit --val-dir wins; otherwise validation/ under the resolved root.
    """
    if val_dir:
        val_dir = os.path.expanduser(val_dir)
        if not os.path.isdir(val_dir):
            raise SystemExit(f"--val-dir not found: {val_dir}")
        return val_dir
    return os.path.join(resolve(root, download=download, quiet=quiet), "validation")


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def _rows(path):
    """Yield the encoded image bytes of a parquet shard, in row order."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    col = "image" if "image" in table.column_names else table.column_names[0]
    for rec in table.column(col).to_pylist():
        yield rec["bytes"] if isinstance(rec, dict) else rec


def _download(dest, quiet=False):
    try:
        import numpy as np
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet  # noqa: F401  (imported for the clear error below)
        from PIL import Image
    except ImportError as e:
        raise SystemExit(
            f"Automatic download needs numpy, huggingface_hub, pyarrow and Pillow ({e}).\n"
            f"    pip install numpy huggingface_hub pyarrow Pillow\n"
            f"    ...or pass --data /path/to/celebahq256 and skip the download.")

    if not os.path.exists(_MANIFEST):
        raise SystemExit(
            f"Cannot rebuild the paper's validation order: {_MANIFEST} is missing.\n"
            f"    The mirror's own train/validation split is NOT the paper's, so\n"
            f"    materialising it would score a different 1,000 images.\n"
            f"    Pass --data /path/to/celebahq256 instead.")

    man = np.load(_MANIFEST)
    M = man["fingerprints"].astype(np.float32)      # (2000, 256), paper order
    if M.shape[0] != N_VAL:
        raise SystemExit(f"manifest holds {M.shape[0]} entries, expected {N_VAL}")
    Mn = (M ** 2).sum(1)

    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    say(f"CelebA-HQ 256: fetching a pinned copy (~3 GB) to {dest}")
    say(f"  repo {REPO}  revision {REVISION[:12]}")
    say(f"  one-off; later runs reuse the cache. Set TENSORMIND_CELEBAHQ to relocate it.")

    shards = [hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION)
              for f in PARQUETS]

    # Pass 1 -- stream every image, keep only the best match per manifest slot.
    # Bytes are not retained: 30,000 JPEGs would be ~3 GB of RAM.
    best_d = np.full(N_VAL, np.inf, dtype=np.float64)
    best_at = np.full(N_VAL, -1, dtype=np.int64)
    pos = 0
    for path, name in zip(shards, PARQUETS):
        chunk, idx = [], []
        for raw in _rows(path):
            chunk.append(fingerprint(Image.open(io.BytesIO(raw))))
            idx.append(pos)
            pos += 1
        C = np.stack(chunk).astype(np.float32)
        d2 = Mn[:, None] - 2.0 * M @ C.T + (C ** 2).sum(1)[None, :]
        j = d2.argmin(1)
        d = np.sqrt(np.maximum(d2[np.arange(N_VAL), j], 0.0))
        take = d < best_d
        best_d[take] = d[take]
        best_at[take] = np.asarray(idx)[j[take]]
        # Deliberately NOT a running match count: the paper's validation images
        # sit almost entirely in the last shard, so a per-shard tally reads as
        # "0/2,000 matched" for most of the run and looks like a failure.
        say(f"  {name}: {pos:,}/{N_TRAIN + N_VAL:,} scanned")

    if pos != N_TRAIN + N_VAL:
        raise SystemExit(f"mirror yielded {pos:,} images, expected {N_TRAIN + N_VAL:,}; "
                         f"nothing written")

    bad = int((best_d >= MATCH_MAX).sum())
    if bad:
        raise SystemExit(
            f"{bad:,} of the paper's {N_VAL:,} validation images could not be matched\n"
            f"    in the mirror (worst distance {best_d.max():.1f}, limit {MATCH_MAX}).\n"
            f"    The evaluation set cannot be rebuilt, so the published numbers would\n"
            f"    not reproduce. Nothing was written. Pass --data with your own copy.")
    if len(set(best_at.tolist())) != N_VAL:
        raise SystemExit(
            f"two validation slots claimed the same source image; the split is\n"
            f"    ambiguous and nothing was written. Pass --data with your own copy.")
    say(f"  matched all {N_VAL:,} validation slots, worst distance {best_d.max():.1f} "
        f"(limit {MATCH_MAX})")

    # Pass 2 -- re-read (now cached locally) and write.
    slot_of = {int(p): i for i, p in enumerate(best_at)}
    tmp = dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ("train", "validation"):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    pos, n_train = 0, 0
    for path in shards:
        for raw in _rows(path):
            slot = slot_of.get(pos)
            if slot is None:
                out = os.path.join(tmp, "train", f"{n_train:06d}.jpg")
                n_train += 1
            else:
                out = os.path.join(tmp, "validation", f"{slot:06d}.jpg")
            # written through unchanged: no re-encode, no second generation of
            # JPEG loss on top of the mirror's.
            with open(out, "wb") as fh:
                fh.write(raw)
            pos += 1

    json.dump({"repo": REPO, "revision": REVISION,
               "manifest": os.path.basename(_MANIFEST),
               "worst_match_distance": round(float(best_d.max()), 3),
               "n_train": n_train, "n_validation": N_VAL},
              open(os.path.join(tmp, "SOURCE.json"), "w"), indent=1)

    if not is_materialised(tmp):
        raise SystemExit(f"assembled copy does not check out:\n      {_describe(tmp)}")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(tmp, dest)
    say(f"CelebA-HQ ready: {dest}")
    return dest


if __name__ == "__main__":
    # `python3 celebahq.py [path]` -- resolve (downloading if needed) and print.
    print(resolve(sys.argv[1] if len(sys.argv) > 1 else None))
