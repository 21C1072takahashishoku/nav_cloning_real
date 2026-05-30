#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check saved 4ch .npy dataset shape and mask values."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Example: ~/shoku_ws/src/nav_cloning/data/real_4ch_short_test/dataset/img")
    parser.add_argument("--max-files", type=int, default=12)
    args = parser.parse_args()

    img_dir = Path(args.dataset_dir).expanduser()
    files = sorted(img_dir.glob("*.npy"))[:args.max_files]
    if not files:
        raise SystemExit(f"No .npy files found: {img_dir}")

    ok = True
    for f in files:
        arr = np.load(f)
        msg = f"{f.name}: shape={arr.shape}, dtype={arr.dtype}"
        if arr.ndim == 3 and arr.shape[-1] == 4:
            mask = arr[:, :, 3]
            msg += f", mask_min={mask.min():.3f}, mask_max={mask.max():.3f}, ground_ratio={(mask > 0.5).mean():.3f}"
        else:
            msg += "  <-- NG: expected HWC=(48,64,4)"
            ok = False
        print(msg)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
