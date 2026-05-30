#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_color_segmentation_legend_mmseg.py

Purpose:
    Create a full semantic-segmentation color visualization and legend from image(s).

Input:
    --raw_input can be either:
        1) one image file: .png / .jpg / .jpeg / .npy
        2) one directory containing image files

Output under out_root:
    seg_color/
        <prefix>_pred_color.png
        <prefix>_overlay_color.png
        <prefix>_overlay_with_legend.png
        <prefix>_legend.png
        <prefix>_legend.csv
        <prefix>_pred_id.npy
        <prefix>_pred_id.png
        summary_color_segmentation.csv

Notes:
    - Color means predicted semantic class.
    - pred_id.npy stores the class ID of every pixel/cell.
    - legend.csv and legend.png explain what each color means.
"""

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


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".npy"}


DEFAULT_CFG = (
    "/home/orne_beta/newsegproject/mmsegmentation/configs/deeplabv3/"
    "deeplabv3_r101-d8_512x512_160k_ade20k.py"
)

DEFAULT_CKPT = (
    "/home/orne_beta/newsegproject/mmsegmentation/checkpoints/"
    "deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create semantic segmentation color maps and class-color legends "
            "from one image or a directory of images."
        )
    )
    parser.add_argument(
        "--raw_input",
        required=True,
        help="Image file or directory. Supports .png, .jpg, .jpeg, .npy.",
    )
    parser.add_argument("--out_root", default=None, help="Output root directory. If omitted, it is inferred from raw_input.")
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=30, help="Number of classes shown in legend.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Original image weight for overlay. 0.0-1.0.")
    parser.add_argument("--recursive", action="store_true", help="Process images in subdirectories too.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument("--save_pred_id_png", action="store_true", help="Also save uint16 class-id PNG.")
    parser.add_argument("--min_h", type=int, default=240, help="If image height is smaller, resize for inference.")
    parser.add_argument("--min_w", type=int, default=320, help="If image width is smaller, resize for inference.")
    parser.add_argument("--infer_h", type=int, default=480, help="Inference resize height for small images.")
    parser.add_argument("--infer_w", type=int, default=640, help="Inference resize width for small images.")
    return parser.parse_args()


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip())


def make_prefix(input_path: Path, raw_root: Path) -> str:
    """Create stable output prefix. Preserves subdir information when recursive."""
    try:
        rel = input_path.relative_to(raw_root)
        stem = str(rel.with_suffix(""))
        return safe_name(stem.replace(os.sep, "__"))
    except Exception:
        return safe_name(input_path.stem)


def collect_images(raw_input: str, recursive: bool = False, max_images=None):
    p = Path(os.path.expanduser(raw_input)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"raw_input not found: {p}")

    if p.is_file():
        if p.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file extension: {p.suffix}")
        return [p], p.parent

    if p.is_dir():
        pattern = "**/*" if recursive else "*"
        files = [x for x in p.glob(pattern) if x.is_file() and x.suffix.lower() in SUPPORTED_EXTS]
        files = sorted(files)
        if max_images is not None:
            files = files[:max_images]
        if len(files) == 0:
            raise RuntimeError(f"No supported images found in: {p}")
        return files, p

    raise RuntimeError(f"raw_input is neither file nor directory: {p}")


def infer_out_root(raw_input: str):
    """
    Infer out_root from common nav_cloning layout:
        .../data/<DATASET>/dataset/raw_img/center/xxx.png
    returns:
        .../data/<DATASET>
    """
    p = Path(os.path.expanduser(raw_input)).resolve()
    base = p.parent if p.is_file() else p
    parts = list(base.parts)

    # Pattern: .../<dataset>/dataset/...
    for i, part in enumerate(parts):
        if part == "dataset" and i > 0:
            return Path(*parts[:i])

    # Pattern: .../data/<dataset>/...
    for i, part in enumerate(parts):
        if part == "data" and i + 1 < len(parts):
            return Path(*parts[: i + 2])

    # Fallback: create output under the input directory/parent
    return base / "segmentation_outputs"


def load_rgb_image(input_path: Path):
    """
    Return:
        rgb_float: float32 HWC, 0.0-1.0
        rgb_u8:    uint8   HWC, 0-255
    """
    ext = input_path.suffix.lower()

    if ext == ".npy":
        x = np.load(str(input_path)).astype(np.float32)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"Unexpected npy image shape: {x.shape} in {input_path}")

        if x.max() <= 1.0:
            rgb_float = x.astype(np.float32)
            rgb_u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb_u8 = x.clip(0, 255).astype(np.uint8)
            rgb_float = rgb_u8.astype(np.float32) / 255.0
        return rgb_float, rgb_u8

    if ext in {".png", ".jpg", ".jpeg"}:
        bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {input_path}")
        rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_float = rgb_u8.astype(np.float32) / 255.0
        return rgb_float, rgb_u8

    raise ValueError(f"Unsupported input extension: {ext}")


def build_model(cfg_path: str, ckpt_path: str, device: str):
    cfg_path = os.path.expanduser(cfg_path)
    ckpt_path = os.path.expanduser(ckpt_path)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    cfg = Config.fromfile(cfg_path)
    if "model" not in cfg:
        raise RuntimeError("cfg.model not found")

    # Guard for old config used with newer MMSeg API.
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
        raise RuntimeError("model.dataset_meta['classes'] not found. config/checkpoint mismatch may exist.")

    return model


@torch.no_grad()
def infer_pred(model, rgb_uint8: np.ndarray) -> np.ndarray:
    """Input HWC RGB uint8. Output HW int32 class-id map."""
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


def get_palette(model, num_classes: int):
    meta = getattr(model, "dataset_meta", {}) or {}
    palette = meta.get("palette", None)

    if palette is not None and len(palette) >= num_classes:
        arr = np.asarray(palette[:num_classes], dtype=np.uint8)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr

    # Deterministic fallback palette in RGB.
    rng = np.random.RandomState(1234)
    arr = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    arr[0] = np.array([0, 0, 0], dtype=np.uint8)
    return arr


def colorize_pred(pred: np.ndarray, palette_rgb: np.ndarray):
    pred_clip = np.clip(pred, 0, len(palette_rgb) - 1).astype(np.int32)
    return palette_rgb[pred_clip]  # RGB HWC


def make_overlay(rgb_u8: np.ndarray, color_rgb: np.ndarray, alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = alpha * rgb_u8.astype(np.float32) + (1.0 - alpha) * color_rgb.astype(np.float32)
    return overlay.clip(0, 255).astype(np.uint8)


def class_histogram(pred: np.ndarray, classes, palette_rgb: np.ndarray):
    ids, counts = np.unique(pred, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = int(pred.size)
    rows = []
    for rank, idx in enumerate(order, start=1):
        cid = int(ids[idx])
        count = int(counts[idx])
        ratio = count / total if total > 0 else 0.0
        cname = classes[cid] if cid < len(classes) else "unknown"
        r, g, b = [int(v) for v in palette_rgb[cid]] if cid < len(palette_rgb) else [0, 0, 0]
        rows.append(
            dict(
                rank=rank,
                class_id=cid,
                class_name=cname,
                count=count,
                ratio=ratio,
                color_r=r,
                color_g=g,
                color_b=b,
                color_hex=f"#{r:02X}{g:02X}{b:02X}",
            )
        )
    return rows


def save_legend_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "class_id", "class_name", "count", "ratio", "color_r", "color_g", "color_b", "color_hex"])
        for row in rows:
            writer.writerow([
                row["rank"], row["class_id"], row["class_name"], row["count"],
                f"{row['ratio']:.8f}", row["color_r"], row["color_g"], row["color_b"], row["color_hex"]
            ])


def make_legend_image(rows, topk: int, title: str = "Color legend"):
    rows = rows[:topk]
    row_h = 34
    header_h = 45
    width = 760
    height = header_h + row_h * max(1, len(rows)) + 15

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    y = header_h
    for row in rows:
        # OpenCV uses BGR.
        color_bgr = (int(row["color_b"]), int(row["color_g"]), int(row["color_r"]))
        cv2.rectangle(img, (12, y - 22), (42, y + 5), color_bgr, thickness=-1)
        cv2.rectangle(img, (12, y - 22), (42, y + 5), (0, 0, 0), thickness=1)

        text = (
            f"{row['rank']:02d}  id={row['class_id']:03d}  "
            f"{row['class_name']}  ratio={row['ratio']:.4f}  "
            f"RGB=({row['color_r']},{row['color_g']},{row['color_b']})"
        )
        cv2.putText(img, text, (55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
        y += row_h

    return img


def combine_overlay_and_legend(overlay_rgb: np.ndarray, legend_bgr: np.ndarray):
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    h1, w1 = overlay_bgr.shape[:2]
    h2, w2 = legend_bgr.shape[:2]

    # Resize legend height to overlay height for readable side-by-side output.
    if h2 != h1:
        scale = h1 / max(1, h2)
        new_w = max(260, int(w2 * scale))
        legend_bgr = cv2.resize(legend_bgr, (new_w, h1), interpolation=cv2.INTER_AREA)

    canvas = np.full((h1, w1 + legend_bgr.shape[1], 3), 255, dtype=np.uint8)
    canvas[:, :w1] = overlay_bgr
    canvas[:, w1:] = legend_bgr
    return canvas


def process_one(input_path: Path, raw_root: Path, out_dir: Path, model, classes, palette_rgb, args):
    prefix = make_prefix(input_path, raw_root)
    print(f"\n===== processing: {input_path} =====")

    rgb_float, rgb_u8 = load_rgb_image(input_path)
    print("rgb_float:", rgb_float.shape, rgb_float.dtype, float(rgb_float.min()), float(rgb_float.max()))
    print("rgb_u8:", rgb_u8.shape, rgb_u8.dtype, int(rgb_u8.min()), int(rgb_u8.max()))

    infer_rgb = rgb_u8
    resized_for_infer = False
    if rgb_u8.shape[0] < args.min_h or rgb_u8.shape[1] < args.min_w:
        infer_rgb = cv2.resize(rgb_u8, (args.infer_w, args.infer_h), interpolation=cv2.INTER_CUBIC)
        resized_for_infer = True

    pred_infer = infer_pred(model, infer_rgb)

    # Return prediction to original image size if inference used resized image.
    if resized_for_infer:
        pred = cv2.resize(pred_infer.astype(np.int32), (rgb_u8.shape[1], rgb_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        pred = pred_infer

    rows = class_histogram(pred, classes, palette_rgb)
    color_rgb = colorize_pred(pred, palette_rgb)
    overlay_rgb = make_overlay(rgb_u8, color_rgb, args.alpha)
    legend_bgr = make_legend_image(rows, args.topk, title=f"Legend: {prefix}")
    combined_bgr = combine_overlay_and_legend(overlay_rgb, legend_bgr)

    pred_color_path = out_dir / f"{prefix}_pred_color.png"
    overlay_path = out_dir / f"{prefix}_overlay_color.png"
    legend_path = out_dir / f"{prefix}_legend.png"
    combined_path = out_dir / f"{prefix}_overlay_with_legend.png"
    legend_csv_path = out_dir / f"{prefix}_legend.csv"
    pred_id_npy_path = out_dir / f"{prefix}_pred_id.npy"
    pred_id_png_path = out_dir / f"{prefix}_pred_id.png"

    cv2.imwrite(str(pred_color_path), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(legend_path), legend_bgr)
    cv2.imwrite(str(combined_path), combined_bgr)
    save_legend_csv(legend_csv_path, rows)
    np.save(str(pred_id_npy_path), pred.astype(np.int32))

    if args.save_pred_id_png:
        cv2.imwrite(str(pred_id_png_path), pred.astype(np.uint16))

    print("top classes:")
    for row in rows[: min(args.topk, 10)]:
        print(
            f"{row['rank']:02d}: id={row['class_id']:3d}, name={row['class_name']:20s}, "
            f"ratio={row['ratio']:.4f}, color={row['color_hex']}"
        )

    print("saved:", pred_color_path)
    print("saved:", overlay_path)
    print("saved:", legend_path)
    print("saved:", combined_path)
    print("saved:", legend_csv_path)
    print("saved:", pred_id_npy_path)

    top1 = rows[0] if rows else None
    return dict(
        input=str(input_path),
        prefix=prefix,
        height=int(rgb_u8.shape[0]),
        width=int(rgb_u8.shape[1]),
        num_classes_detected=len(rows),
        top1_class_id=top1["class_id"] if top1 else -1,
        top1_class_name=top1["class_name"] if top1 else "",
        top1_ratio=top1["ratio"] if top1 else 0.0,
        overlay=str(overlay_path),
        legend_csv=str(legend_csv_path),
    )


def main():
    args = parse_args()

    images, raw_root = collect_images(args.raw_input, recursive=args.recursive, max_images=args.max_images)
    out_root = Path(os.path.expanduser(args.out_root)).resolve() if args.out_root else infer_out_root(args.raw_input)
    out_dir = out_root / "seg_color/class_visualization/"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("raw_input:", os.path.expanduser(args.raw_input))
    print("num_images:", len(images))
    print("out_root:", out_root)
    print("seg_color_dir:", out_dir)

    model = build_model(args.cfg, args.ckpt, args.device)
    classes = list(model.dataset_meta["classes"])
    palette_rgb = get_palette(model, len(classes))

    summary_rows = []
    for image_path in images:
        try:
            summary_rows.append(process_one(image_path, raw_root, out_dir, model, classes, palette_rgb, args))
        except Exception as e:
            print(f"ERROR: failed to process {image_path}: {e}")
            summary_rows.append(
                dict(
                    input=str(image_path),
                    prefix=make_prefix(image_path, raw_root),
                    height=-1,
                    width=-1,
                    num_classes_detected=-1,
                    top1_class_id=-1,
                    top1_class_name="ERROR",
                    top1_ratio=0.0,
                    overlay="",
                    legend_csv="",
                )
            )

    summary_path = out_dir / "summary_color_segmentation.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "input", "prefix", "height", "width", "num_classes_detected",
            "top1_class_id", "top1_class_name", "top1_ratio", "overlay", "legend_csv"
        ])
        for row in summary_rows:
            writer.writerow([
                row["input"], row["prefix"], row["height"], row["width"], row["num_classes_detected"],
                row["top1_class_id"], row["top1_class_name"], f"{row['top1_ratio']:.8f}",
                row["overlay"], row["legend_csv"]
            ])

    print("\n===== done =====")
    print("summary:", summary_path)
    print("output:", out_dir)


if __name__ == "__main__":
    main()
