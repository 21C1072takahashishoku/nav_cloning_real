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
            "Diagnose MMSegmentation prediction classes. "
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
    parser.add_argument("--topk", type=int, default=20)
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

    # Standard nav_cloning path:
    # /home/orne_beta/shoku_ws/src/nav_cloning/data/<dataset>/dataset/raw_img/center/0_center.png
    data_indices = [i for i, part in enumerate(parts) if part == "data"]
    for i in reversed(data_indices):
        if i + 1 < len(parts):
            return str(Path(*parts[: i + 2]))

    base = p if os.path.isdir(p) else os.path.dirname(p)
    return os.path.join(base, "_mmseg_output")


def load_rgb_image(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy":
        rgb = np.load(input_path).astype(np.float32)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Unexpected npy image shape: {rgb.shape}")

        if rgb.max() <= 1.0:
            rgb_float = rgb
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

    raise ValueError(f"Unsupported extension: {ext}")


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
        raise RuntimeError("model.dataset_meta['classes'] not found.")

    return model


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


def process_one(input_path, model, classes, debug_dir, topk):
    prefix = safe_name(os.path.splitext(os.path.basename(input_path))[0])

    print(f"\n===== processing: {input_path} =====")

    rgb_float, rgb_u8 = load_rgb_image(input_path)
    print("rgb_float:", rgb_float.shape, rgb_float.dtype, rgb_float.min(), rgb_float.max())
    print("rgb_u8:", rgb_u8.shape, rgb_u8.dtype, rgb_u8.min(), rgb_u8.max())

    infer_rgb = rgb_u8
    if rgb_u8.shape[0] < 240 or rgb_u8.shape[1] < 320:
        infer_rgb = cv2.resize(rgb_u8, (640, 480), interpolation=cv2.INTER_CUBIC)

    pred = infer_pred(model, infer_rgb)

    pred_ids, pred_counts = np.unique(pred, return_counts=True)
    order = np.argsort(pred_counts)[::-1]
    total = pred.size

    print(f"===== predicted class histogram top {topk} =====")

    hist_rows = []
    for rank, idx in enumerate(order[:topk], start=1):
        class_id = int(pred_ids[idx])
        count = int(pred_counts[idx])
        ratio = count / total
        class_name = classes[class_id] if class_id < len(classes) else "unknown"

        print(
            f"{rank:02d}: id={class_id:3d}, "
            f"name={class_name:20s}, count={count:8d}, ratio={ratio:.4f}"
        )

        hist_rows.append([rank, class_id, class_name, count, ratio])

    csv_path = os.path.join(debug_dir, f"{prefix}_pred_hist_top{topk}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "class_id", "class_name", "count", "ratio"])
        writer.writerows(hist_rows)

    input_png = os.path.join(debug_dir, f"{prefix}_input.png")
    cv2.imwrite(input_png, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

    for rank, class_id, class_name, count, ratio in hist_rows[:10]:
        class_mask_big = (pred == class_id).astype(np.uint8)
        class_mask_small = cv2.resize(
            class_mask_big,
            (rgb_u8.shape[1], rgb_u8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_path = os.path.join(
            debug_dir,
            f"{prefix}_top{rank:02d}_{class_id:03d}_{safe_name(class_name)}.png",
        )
        cv2.imwrite(mask_path, class_mask_small * 255)

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

    name2id = {str(name).lower(): i for i, name in enumerate(classes)}
    ground_ids = [name2id[name] for name in ground_names if name in name2id]
    ground_ratio = float(np.mean(np.isin(pred, ground_ids)))

    print("ground_ids:", ground_ids)
    print("ground_ratio on infer image:", ground_ratio)
    print("saved:", csv_path)
    print("saved:", input_png)

    top1_id = hist_rows[0][1] if hist_rows else None
    top1_name = hist_rows[0][2] if hist_rows else None
    top1_ratio = hist_rows[0][4] if hist_rows else None

    return {
        "input_path": input_path,
        "prefix": prefix,
        "status": "ok",
        "top1_id": top1_id,
        "top1_name": top1_name,
        "top1_ratio": top1_ratio,
        "ground_ratio": ground_ratio,
        "csv_path": csv_path,
    }


def main():
    args = parse_args()

    image_paths = collect_inputs(args.raw_input, recursive=args.recursive)
    out_root = infer_out_root(args.raw_input, args.out_root)
    debug_dir = os.path.join(out_root, "seg_debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("raw_input:", expand_path(args.raw_input))
    print("num_images:", len(image_paths))
    print("out_root:", out_root)
    print("debug_dir:", debug_dir)

    model = build_model(args.cfg, args.ckpt, args.device)
    classes = list(model.dataset_meta["classes"])

    summary_rows = []

    for input_path in image_paths:
        try:
            result = process_one(input_path, model, classes, debug_dir, args.topk)
            summary_rows.append(result)
        except Exception as e:
            print(f"[ERROR] {input_path}: {e}")
            summary_rows.append(
                {
                    "input_path": input_path,
                    "prefix": safe_name(os.path.splitext(os.path.basename(input_path))[0]),
                    "status": "error",
                    "top1_id": "",
                    "top1_name": "",
                    "top1_ratio": "",
                    "ground_ratio": "",
                    "csv_path": "",
                    "error": str(e),
                }
            )
            if args.stop_on_error:
                raise

    summary_csv = os.path.join(debug_dir, "summary_prediction.csv")
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "input_path",
            "prefix",
            "status",
            "top1_id",
            "top1_name",
            "top1_ratio",
            "ground_ratio",
            "csv_path",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("\n===== done =====")
    print("summary:", summary_csv)
    print("output:", debug_dir)


if __name__ == "__main__":
    main()
