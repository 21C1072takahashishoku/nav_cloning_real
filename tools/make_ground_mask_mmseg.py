#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from mmengine.config import Config
from mmseg.apis import init_model
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample


DEFAULT_CFG = "/home/orne_beta/newsegproject/mmsegmentation/configs/deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k.py"
DEFAULT_CKPT = "/home/orne_beta/newsegproject/mmsegmentation/checkpoints/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create ground mask and 4ch input from nav_cloning images. "
            "--raw_input can be one image file or one directory."
        )
    )
    parser.add_argument(
        "--raw_input",
        required=True,
        help="Input image file or directory. Supports .png, .jpg, .jpeg, .npy.",
    )
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-h", type=int, default=48)
    parser.add_argument("--out-w", type=int, default=64)
    parser.add_argument(
        "--out_root",
        default=None,
        help=(
            "Output root directory. If omitted, the script tries to infer "
            ".../data/<dataset> from --raw_input."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --raw_input is a directory, also search subdirectories.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when one image fails.",
    )
    return parser.parse_args()


def expand_path(path):
    return os.path.abspath(os.path.expanduser(str(path)))


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip())


def natural_key(path):
    name = os.path.basename(str(path))
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", name)]


def collect_inputs(raw_input, recursive=False):
    raw_input = expand_path(raw_input)

    if not os.path.exists(raw_input):
        raise FileNotFoundError(f"raw_input not found: {raw_input}")

    if os.path.isfile(raw_input):
        ext = os.path.splitext(raw_input)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file extension: {ext}")
        return [raw_input]

    if not os.path.isdir(raw_input):
        raise ValueError(f"raw_input is neither file nor directory: {raw_input}")

    image_paths = []
    if recursive:
        for root, _, files in os.walk(raw_input):
            for f in files:
                p = os.path.join(root, f)
                if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                    image_paths.append(p)
    else:
        for f in os.listdir(raw_input):
            p = os.path.join(raw_input, f)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                image_paths.append(p)

    image_paths = sorted(image_paths, key=natural_key)

    if len(image_paths) == 0:
        raise RuntimeError(f"No supported images found in: {raw_input}")

    return image_paths


def infer_out_root(raw_input, out_root=None):
    if out_root is not None:
        return expand_path(out_root)

    p = expand_path(raw_input)
    parts = Path(p).parts

    data_indices = [i for i, part in enumerate(parts) if part == "data"]
    for i in reversed(data_indices):
        if i + 1 < len(parts):
            return str(Path(*parts[: i + 2]))

    base = p if os.path.isdir(p) else os.path.dirname(p)
    return os.path.join(base, "_mmseg_output")


def load_rgb_image(input_path):
    """
    Return:
        rgb_float: float32, HWC, 0.0-1.0
        rgb_u8:    uint8, HWC, 0-255
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy":
        rgb = np.load(input_path).astype(np.float32)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Unexpected npy image shape: {rgb.shape}")

        if rgb.max() <= 1.0:
            rgb_float = rgb.astype(np.float32)
            rgb_u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb_u8 = rgb.clip(0, 255).astype(np.uint8)
            rgb_float = rgb_u8.astype(np.float32) / 255.0

        return rgb_float, rgb_u8

    if ext in [".png", ".jpg", ".jpeg"]:
        bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)

        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {input_path}")

        rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_float = rgb_u8.astype(np.float32) / 255.0

        return rgb_float, rgb_u8

    raise ValueError(f"Unsupported input extension: {ext}")


def build_model(cfg_path, ckpt_path, device):
    cfg_path = expand_path(cfg_path)
    ckpt_path = expand_path(ckpt_path)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Use CPU instead.")
        device = "cpu"

    cfg = Config.fromfile(cfg_path)

    if "model" not in cfg:
        raise RuntimeError("cfg.model not found")

    if "data_preprocessor" not in cfg.model:
        cfg.model.data_preprocessor = dict(
            type="SegDataPreProcessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255,
        )

    model = init_model(cfg, ckpt_path, device=device)
    model.eval()

    if not (
        hasattr(model, "dataset_meta")
        and model.dataset_meta
        and "classes" in model.dataset_meta
    ):
        raise RuntimeError(
            "model.dataset_meta['classes'] not found. config/checkpoint mismatch may exist."
        )

    return model


def get_ground_ids(model):
    classes = list(model.dataset_meta["classes"])
    name2id = {str(name).lower(): idx for idx, name in enumerate(classes)}

    ground_names = {
        "floor",
        "road",
        "sidewalk",
        "path",
        "earth",
        "land",
        "field",
        "sand",
        "grass",
        "runway",
    }

    ground_ids = []
    unknown = []

    for name in sorted(ground_names):
        if name in name2id:
            ground_ids.append(name2id[name])
        else:
            unknown.append(name)

    print("ground_ids:", ground_ids)
    print("unknown ground names:", unknown)

    if len(ground_ids) == 0:
        raise RuntimeError("ground_ids is empty. Check ground_names and model dataset classes.")

    return np.array(ground_ids, dtype=np.int32)


@torch.no_grad()
def infer_pred(model, rgb_uint8):
    h, w = rgb_uint8.shape[:2]

    sample = SegDataSample()
    sample.set_metainfo(
        dict(
            ori_shape=(h, w),
            img_shape=(h, w),
            pad_shape=(h, w),
            scale_factor=1.0,
        )
    )

    inp = torch.from_numpy(rgb_uint8).permute(2, 0, 1)

    data = dict(inputs=[inp], data_samples=[sample])
    data = model.data_preprocessor(data, training=False)
    outputs = model.test_step(data)

    pred_sem = outputs[0].pred_sem_seg
    pred = pred_sem.data if isinstance(pred_sem, PixelData) else pred_sem

    if hasattr(pred, "dim") and callable(getattr(pred, "dim")):
        if pred.dim() == 3:
            pred = pred[0]
        pred = pred.detach().cpu().numpy()
    else:
        if getattr(pred, "ndim", None) == 3:
            pred = pred[0]

    return pred.astype(np.int32)


def process_one(input_path, model, ground_ids, dirs, out_h, out_w):
    seg_output_dir, seg_vis_dir, check_4ch_dir = dirs
    prefix = safe_name(os.path.splitext(os.path.basename(input_path))[0])

    print(f"\n===== processing: {input_path} =====")

    rgb_float, rgb_u8 = load_rgb_image(input_path)
    print("rgb_float:", rgb_float.shape, rgb_float.dtype, rgb_float.min(), rgb_float.max())
    print("rgb_u8:", rgb_u8.shape, rgb_u8.dtype, rgb_u8.min(), rgb_u8.max())

    infer_rgb = rgb_u8
    if rgb_u8.shape[0] < 240 or rgb_u8.shape[1] < 320:
        infer_rgb = cv2.resize(rgb_u8, (640, 480), interpolation=cv2.INTER_CUBIC)

    pred = infer_pred(model, infer_rgb)

    is_ground_full = np.isin(pred, ground_ids)
    mask_full_01 = is_ground_full.astype(np.uint8)

    mask_small_01 = cv2.resize(
        mask_full_01,
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)

    rgb_small_u8 = cv2.resize(
        rgb_u8,
        (out_w, out_h),
        interpolation=cv2.INTER_AREA,
    )

    rgb_small_float = rgb_small_u8.astype(np.float32) / 255.0

    ground_ratio = float(np.mean(mask_small_01 == 1))
    non_ground_ratio = float(np.mean(mask_small_01 == 0))
    unique_values = np.unique(mask_small_01)

    print("mask shape:", mask_small_01.shape)
    print("mask unique:", unique_values)
    print(f"ground_ratio={ground_ratio:.3f}, non_ground_ratio={non_ground_ratio:.3f}")

    if ground_ratio < 0.01:
        print("WARNING: almost no ground detected.")
    if ground_ratio > 0.99:
        print("WARNING: almost all pixels are ground.")

    mask_npy_path = os.path.join(seg_output_dir, f"{prefix}_ground_mask.npy")
    np.save(mask_npy_path, mask_small_01.astype(np.float32))

    input_png_path = os.path.join(seg_vis_dir, f"{prefix}_input.png")
    mask_png_path = os.path.join(seg_vis_dir, f"{prefix}_ground_mask.png")
    overlay_png_path = os.path.join(seg_vis_dir, f"{prefix}_overlay.png")

    cv2.imwrite(input_png_path, cv2.cvtColor(rgb_small_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_png_path, mask_small_01 * 255)

    overlay = rgb_small_u8.copy()
    overlay[mask_small_01 == 1] = (
        0.5 * overlay[mask_small_01 == 1]
        + 0.5 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    cv2.imwrite(overlay_png_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    mask_hwc = mask_small_01[:, :, None].astype(np.float32)

    x4_hwc = np.concatenate(
        [rgb_small_float.astype(np.float32), mask_hwc],
        axis=2,
    )

    r = rgb_small_u8[:, :, 0].astype(np.float32)
    g = rgb_small_u8[:, :, 1].astype(np.float32)
    b = rgb_small_u8[:, :, 2].astype(np.float32)
    mask01 = mask_small_01.astype(np.float32)

    x4_chw = np.asarray([r, g, b, mask01], dtype=np.float32)

    x4_hwc_path = os.path.join(check_4ch_dir, f"{prefix}_4ch_hwc.npy")
    x4_chw_path = os.path.join(check_4ch_dir, f"{prefix}_4ch_chw.npy")

    np.save(x4_hwc_path, x4_hwc)
    np.save(x4_chw_path, x4_chw)

    print("saved:", mask_npy_path)
    print("saved:", input_png_path)
    print("saved:", mask_png_path)
    print("saved:", overlay_png_path)
    print("saved:", x4_hwc_path, x4_hwc.shape, x4_hwc.dtype, x4_hwc.min(), x4_hwc.max())
    print("saved:", x4_chw_path, x4_chw.shape, x4_chw.dtype, x4_chw.min(), x4_chw.max())

    return {
        "input_path": input_path,
        "prefix": prefix,
        "status": "ok",
        "mask_unique": " ".join(map(str, unique_values.tolist())),
        "ground_ratio": ground_ratio,
        "non_ground_ratio": non_ground_ratio,
        "mask_npy_path": mask_npy_path,
        "overlay_png_path": overlay_png_path,
        "x4_hwc_path": x4_hwc_path,
        "x4_chw_path": x4_chw_path,
    }


def main():
    args = parse_args()

    image_paths = collect_inputs(args.raw_input, recursive=args.recursive)
    out_root = infer_out_root(args.raw_input, args.out_root)

    seg_output_dir = os.path.join(out_root, "seg_output")
    seg_vis_dir = os.path.join(out_root, "seg_vis")
    check_4ch_dir = os.path.join(out_root, "check_4ch")

    for d in [seg_output_dir, seg_vis_dir, check_4ch_dir]:
        os.makedirs(d, exist_ok=True)

    print("raw_input:", expand_path(args.raw_input))
    print("num_images:", len(image_paths))
    print("out_root:", out_root)
    print("seg_output_dir:", seg_output_dir)
    print("seg_vis_dir:", seg_vis_dir)
    print("check_4ch_dir:", check_4ch_dir)

    model = build_model(args.cfg, args.ckpt, args.device)
    ground_ids = get_ground_ids(model)

    summary_rows = []
    dirs = (seg_output_dir, seg_vis_dir, check_4ch_dir)

    for input_path in image_paths:
        try:
            result = process_one(
                input_path=input_path,
                model=model,
                ground_ids=ground_ids,
                dirs=dirs,
                out_h=args.out_h,
                out_w=args.out_w,
            )
            summary_rows.append(result)
        except Exception as e:
            print(f"[ERROR] {input_path}: {e}")
            summary_rows.append(
                {
                    "input_path": input_path,
                    "prefix": safe_name(os.path.splitext(os.path.basename(input_path))[0]),
                    "status": "error",
                    "mask_unique": "",
                    "ground_ratio": "",
                    "non_ground_ratio": "",
                    "mask_npy_path": "",
                    "overlay_png_path": "",
                    "x4_hwc_path": "",
                    "x4_chw_path": "",
                    "error": str(e),
                }
            )
            if args.stop_on_error:
                raise

    summary_csv = os.path.join(seg_output_dir, "summary_ground_mask.csv")
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "input_path",
            "prefix",
            "status",
            "mask_unique",
            "ground_ratio",
            "non_ground_ratio",
            "mask_npy_path",
            "overlay_png_path",
            "x4_hwc_path",
            "x4_chw_path",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("\n===== done =====")
    print("summary:", summary_csv)
    print("output:", out_root)


if __name__ == "__main__":
    main()
