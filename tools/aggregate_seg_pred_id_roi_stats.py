#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate MMSegmentation prediction ID maps in a region of interest (ROI).

Purpose:
    This script calculates class ratios and ground ratios only inside a selected ROI,
    such as the lower half of the image.

Important:
    Existing *_legend.csv files contain whole-image class ratios only.
    They do not contain pixel position information.
    Therefore, ROI statistics must be calculated from *_pred_id.npy.

Input:
    --seg_color_dir:
        Root directory of color segmentation outputs.
        Example:
            /home/orne_beta/shoku_ws/src/nav_cloning/data/train_test_3ch_full_4lap_001/seg_color

    The script automatically reads prediction files from:
        <seg_color_dir>/<class_visualization_subdir>/

    Default class_visualization_subdir:
        class_visualization

    Required files in class_visualization:
        *_pred_id.npy

    Optional files in class_visualization:
        *_legend.csv

Output:
    --out_dir:
        Output directory specified by user.

Main outputs:
    summary_roi_input_files.csv
    roi_class_ratio_by_image_long.csv
    roi_class_ratio_by_image_wide.csv
    roi_class_average_overall.csv
    roi_class_average_by_group_<N>.csv
    roi_ground_ratio_by_image.csv
    roi_ground_ratio_by_group_<N>.csv

Default ROI:
    lower half of prediction image:
        y_start_ratio = 0.5
        y_end_ratio   = 1.0
"""

import os
import re
import csv
import math
import argparse
from pathlib import Path

import numpy as np


DEFAULT_GROUND_NAMES = "floor,road,sidewalk,path,earth,land,field,sand,grass,runway"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate *_pred_id.npy segmentation maps over bottom-half or custom ROI."
    )

    parser.add_argument(
        "--seg_color_dir",
        required=True,
        help="Root directory of segmentation color outputs, e.g. .../seg_color",
    )

    parser.add_argument(
        "--class_visualization_subdir",
        default="class_visualization",
        help="Subdirectory under seg_color_dir containing *_pred_id.npy. default: class_visualization",
    )

    parser.add_argument(
        "--pred_pattern",
        default="*_pred_id.npy",
        help="Prediction ID filename pattern. default: *_pred_id.npy",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for ROI statistics.",
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=100,
        help="Number of images per group. default: 100",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index after sorting files. 0-based. default: 0",
    )

    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index after sorting files. exclusive. default: all",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search class_visualization directory recursively.",
    )

    parser.add_argument(
        "--y_start_ratio",
        type=float,
        default=0.5,
        help="ROI start y ratio. 0.5 means lower half starts at image middle. default: 0.5",
    )

    parser.add_argument(
        "--y_end_ratio",
        type=float,
        default=1.0,
        help="ROI end y ratio. 1.0 means image bottom. default: 1.0",
    )

    parser.add_argument(
        "--x_start_ratio",
        type=float,
        default=0.0,
        help="ROI start x ratio. default: 0.0",
    )

    parser.add_argument(
        "--x_end_ratio",
        type=float,
        default=1.0,
        help="ROI end x ratio. default: 1.0",
    )

    parser.add_argument(
        "--ground_names",
        default=DEFAULT_GROUND_NAMES,
        help=(
            "Comma-separated class names treated as ground. "
            "default: floor,road,sidewalk,path,earth,land,field,sand,grass,runway"
        ),
    )

    parser.add_argument(
        "--ground_ids",
        default=None,
        help="Optional comma-separated class IDs treated as ground. If set, this is merged with ground_names.",
    )

    return parser.parse_args()


def natural_key(path: Path):
    """
    Natural sort key.

    Example:
        2_center_pred_id.npy comes before 10_center_pred_id.npy
    """
    s = path.name
    parts = re.split(r"(\d+)", s)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def prefix_from_pred(path: Path):
    """
    Convert:
        1167_center_pred_id.npy
    to:
        1167_center
    """
    name = path.name

    if name.endswith("_pred_id.npy"):
        return name[: -len("_pred_id.npy")]

    return path.stem


def legend_path_from_prefix(class_visualization_dir: Path, prefix: str):
    """
    Return matching legend CSV path.

    Example:
        class_visualization/1167_center_legend.csv
    """
    return class_visualization_dir / f"{prefix}_legend.csv"


def read_legend_meta(path: Path):
    """
    Read class metadata from *_legend.csv.

    Returns:
        meta[class_id] = {
            class_name,
            color_r,
            color_g,
            color_b,
            color_hex
        }
    """
    meta = {}

    if not path.exists():
        return meta

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cid = safe_int(row.get("class_id"))
            meta[cid] = {
                "class_name": row.get("class_name", f"id_{cid}"),
                "color_r": row.get("color_r", ""),
                "color_g": row.get("color_g", ""),
                "color_b": row.get("color_b", ""),
                "color_hex": row.get("color_hex", ""),
            }

    return meta


def load_pred_id(path: Path):
    """
    Load *_pred_id.npy.

    Accepted shapes:
        (H, W)
        (1, H, W)
        (H, W, 1)

    Returns:
        2D int32 array.
    """
    arr = np.load(str(path))
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"{path}: expected 2D pred_id map, got shape={arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D pred_id map, got shape={arr.shape}")

    return arr.astype(np.int32, copy=False)


def clamp_ratio(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def roi_bounds(h, w, args):
    """
    Calculate ROI pixel bounds from ratios.

    y_start_ratio=0.5, y_end_ratio=1.0
        -> lower half
    """
    ys = int(round(h * clamp_ratio(args.y_start_ratio)))
    ye = int(round(h * clamp_ratio(args.y_end_ratio)))
    xs = int(round(w * clamp_ratio(args.x_start_ratio)))
    xe = int(round(w * clamp_ratio(args.x_end_ratio)))

    ys = max(0, min(h, ys))
    ye = max(0, min(h, ye))
    xs = max(0, min(w, xs))
    xe = max(0, min(w, xe))

    if ye <= ys:
        raise ValueError(f"Invalid y ROI: y_start={ys}, y_end={ye}, h={h}")

    if xe <= xs:
        raise ValueError(f"Invalid x ROI: x_start={xs}, x_end={xe}, w={w}")

    return ys, ye, xs, xe


def get_meta_value(meta, cid, key):
    if cid in meta:
        return meta[cid].get(key, "")

    if key == "class_name":
        return f"id_{cid}"

    if key in ("color_r", "color_g", "color_b", "color_hex"):
        return ""

    return ""


def aggregate_class_rows(records, class_ids, meta, group_label="overall_roi"):
    """
    Aggregate class ratio statistics over records.
    """
    total_pixels_all = sum(r["roi_pixels"] for r in records)
    n_images = len(records)
    rows = []

    for cid in class_ids:
        ratios = []
        counts = []
        appeared_images = 0

        for rec in records:
            count = rec["counts"].get(cid, 0)
            ratio = count / rec["roi_pixels"] if rec["roi_pixels"] > 0 else 0.0

            counts.append(count)
            ratios.append(ratio)

            if count > 0:
                appeared_images += 1

        count_sum = sum(counts)
        pixel_ratio = count_sum / total_pixels_all if total_pixels_all > 0 else 0.0

        rows.append([
            group_label,
            n_images,
            cid,
            get_meta_value(meta, cid, "class_name"),
            count_sum,
            f"{pixel_ratio:.10f}",
            f"{mean(ratios):.10f}",
            f"{std(ratios):.10f}",
            f"{min(ratios) if ratios else 0.0:.10f}",
            f"{max(ratios) if ratios else 0.0:.10f}",
            appeared_images,
            get_meta_value(meta, cid, "color_r"),
            get_meta_value(meta, cid, "color_g"),
            get_meta_value(meta, cid, "color_b"),
            get_meta_value(meta, cid, "color_hex"),
        ])

    rows.sort(key=lambda x: float(x[6]), reverse=True)
    return rows


def main():
    args = parse_args()

    seg_color_dir = Path(os.path.expanduser(args.seg_color_dir)).resolve()

    if not seg_color_dir.exists():
        raise FileNotFoundError(f"seg_color_dir not found: {seg_color_dir}")

    class_visualization_subdir = args.class_visualization_subdir.strip().strip("/")
    if not class_visualization_subdir:
        raise ValueError("--class_visualization_subdir must not be empty")

    class_visualization_dir = seg_color_dir / class_visualization_subdir

    if not class_visualization_dir.exists():
        raise FileNotFoundError(f"class_visualization directory not found: {class_visualization_dir}")

    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Collect pred_id files from <seg_color_dir>/<class_visualization_subdir>
    # ------------------------------------------------------------
    globber = class_visualization_dir.rglob if args.recursive else class_visualization_dir.glob

    files = sorted(
        [p for p in globber(args.pred_pattern) if p.is_file()],
        key=natural_key,
    )

    if args.end_index is None:
        selected = files[args.start_index:]
    else:
        selected = files[args.start_index:args.end_index]

    if not selected:
        raise RuntimeError(
            f"No pred_id files found. class_visualization_dir={class_visualization_dir}, pattern={args.pred_pattern}"
        )

    records = []
    meta = {}

    # ------------------------------------------------------------
    # Read all prediction maps and calculate ROI class counts
    # ------------------------------------------------------------
    for local_idx, path in enumerate(selected):
        prefix = prefix_from_pred(path)

        pred = load_pred_id(path)
        h, w = pred.shape

        ys, ye, xs, xe = roi_bounds(h, w, args)

        roi = pred[ys:ye, xs:xe]

        values, counts = np.unique(roi, return_counts=True)
        count_dict = {int(v): int(c) for v, c in zip(values, counts)}

        roi_pixels = int(roi.size)
        full_pixels = int(pred.size)

        # Read class metadata from matching legend CSV, if available.
        local_meta = read_legend_meta(legend_path_from_prefix(class_visualization_dir, prefix))

        for cid, meta_row in local_meta.items():
            if cid not in meta:
                meta[cid] = meta_row

        # Add fallback metadata for class IDs found only in pred_id.
        for cid in count_dict:
            if cid not in meta:
                meta[cid] = {
                    "class_name": f"id_{cid}",
                    "color_r": "",
                    "color_g": "",
                    "color_b": "",
                    "color_hex": "",
                }

        top_cid = max(count_dict, key=count_dict.get) if count_dict else ""
        top_ratio = count_dict.get(top_cid, 0) / roi_pixels if roi_pixels else 0.0

        records.append({
            "local_index": local_idx,
            "file_index": args.start_index + local_idx,
            "file": str(path),
            "file_name": path.name,
            "prefix": prefix,
            "height": h,
            "width": w,
            "y_start": ys,
            "y_end": ye,
            "x_start": xs,
            "x_end": xe,
            "roi_pixels": roi_pixels,
            "full_pixels": full_pixels,
            "top_class_id": top_cid,
            "top_class_name": get_meta_value(meta, top_cid, "class_name") if top_cid != "" else "",
            "top_ratio": top_ratio,
            "counts": count_dict,
        })

    class_ids = sorted(meta.keys())

    # ------------------------------------------------------------
    # 1. Input summary
    # ------------------------------------------------------------
    write_csv(
        out_dir / "summary_roi_input_files.csv",
        [
            "local_index",
            "file_index",
            "file_name",
            "prefix",
            "height",
            "width",
            "y_start",
            "y_end",
            "x_start",
            "x_end",
            "roi_pixels",
            "full_pixels",
            "top_class_id",
            "top_class_name",
            "top_ratio",
            "file",
        ],
        [[
            r["local_index"],
            r["file_index"],
            r["file_name"],
            r["prefix"],
            r["height"],
            r["width"],
            r["y_start"],
            r["y_end"],
            r["x_start"],
            r["x_end"],
            r["roi_pixels"],
            r["full_pixels"],
            r["top_class_id"],
            r["top_class_name"],
            f"{r['top_ratio']:.10f}",
            r["file"],
        ] for r in records],
    )

    # ------------------------------------------------------------
    # 2. Per-image long table
    # ------------------------------------------------------------
    long_rows = []

    for r in records:
        for cid in class_ids:
            count = r["counts"].get(cid, 0)
            ratio = count / r["roi_pixels"] if r["roi_pixels"] else 0.0

            long_rows.append([
                r["local_index"],
                r["file_index"],
                r["file_name"],
                r["prefix"],
                cid,
                get_meta_value(meta, cid, "class_name"),
                count,
                f"{ratio:.10f}",
                get_meta_value(meta, cid, "color_hex"),
            ])

    write_csv(
        out_dir / "roi_class_ratio_by_image_long.csv",
        [
            "local_index",
            "file_index",
            "file_name",
            "prefix",
            "class_id",
            "class_name",
            "count",
            "ratio",
            "color_hex",
        ],
        long_rows,
    )

    # ------------------------------------------------------------
    # 3. Per-image wide table
    # ------------------------------------------------------------
    wide_header = [
        "local_index",
        "file_index",
        "file_name",
        "prefix",
        "roi_pixels",
    ]

    wide_header += [
        f"ratio_id{cid}_{get_meta_value(meta, cid, 'class_name')}"
        for cid in class_ids
    ]

    wide_rows = []

    for r in records:
        row = [
            r["local_index"],
            r["file_index"],
            r["file_name"],
            r["prefix"],
            r["roi_pixels"],
        ]

        for cid in class_ids:
            count = r["counts"].get(cid, 0)
            ratio = count / r["roi_pixels"] if r["roi_pixels"] else 0.0
            row.append(f"{ratio:.10f}")

        wide_rows.append(row)

    write_csv(
        out_dir / "roi_class_ratio_by_image_wide.csv",
        wide_header,
        wide_rows,
    )

    # ------------------------------------------------------------
    # 4. Overall average
    # ------------------------------------------------------------
    avg_header = [
        "group",
        "n_images",
        "class_id",
        "class_name",
        "count_sum",
        "pixel_ratio",
        "mean_ratio",
        "std_ratio",
        "min_ratio",
        "max_ratio",
        "appeared_images",
        "color_r",
        "color_g",
        "color_b",
        "color_hex",
    ]

    write_csv(
        out_dir / "roi_class_average_overall.csv",
        avg_header,
        aggregate_class_rows(
            records,
            class_ids,
            meta,
            "overall_roi",
        ),
    )

    # ------------------------------------------------------------
    # 5. Group average
    # ------------------------------------------------------------
    group_size = max(1, int(args.group_size))

    group_rows = []
    group_info_rows = []

    for group_id, start in enumerate(range(0, len(records), group_size)):
        group_records = records[start:start + group_size]
        end = start + len(group_records) - 1

        label = f"group_{group_id:04d}_images_{start:06d}_{end:06d}_roi"

        group_rows.extend(
            aggregate_class_rows(
                group_records,
                class_ids,
                meta,
                label,
            )
        )

        group_info_rows.append([
            label,
            group_id,
            start,
            end,
            len(group_records),
            group_records[0]["file_name"],
            group_records[-1]["file_name"],
        ])

    write_csv(
        out_dir / f"roi_class_average_by_group_{group_size}.csv",
        avg_header,
        group_rows,
    )

    write_csv(
        out_dir / f"roi_group_info_{group_size}.csv",
        [
            "group",
            "group_id",
            "start_local_index",
            "end_local_index",
            "n_images",
            "start_file",
            "end_file",
        ],
        group_info_rows,
    )

    # ------------------------------------------------------------
    # 6. Ground ratio summaries
    # ------------------------------------------------------------
    ground_names = {
        x.strip().lower()
        for x in args.ground_names.split(",")
        if x.strip()
    }

    ground_ids_by_name = {
        cid
        for cid in class_ids
        if str(get_meta_value(meta, cid, "class_name")).lower() in ground_names
    }

    ground_ids_by_arg = set()

    if args.ground_ids:
        ground_ids_by_arg = {
            safe_int(x)
            for x in args.ground_ids.split(",")
            if x.strip()
        }

    ground_ids = sorted(ground_ids_by_name | ground_ids_by_arg)

    ground_by_image = []

    for r in records:
        ground_count = sum(r["counts"].get(cid, 0) for cid in ground_ids)
        ground_ratio = ground_count / r["roi_pixels"] if r["roi_pixels"] else 0.0

        ground_by_image.append([
            r["local_index"],
            r["file_index"],
            r["file_name"],
            r["prefix"],
            ground_count,
            f"{ground_ratio:.10f}",
            r["roi_pixels"],
            r["y_start"],
            r["y_end"],
            r["x_start"],
            r["x_end"],
        ])

    write_csv(
        out_dir / "roi_ground_ratio_by_image.csv",
        [
            "local_index",
            "file_index",
            "file_name",
            "prefix",
            "ground_count",
            "ground_ratio",
            "roi_pixels",
            "y_start",
            "y_end",
            "x_start",
            "x_end",
        ],
        ground_by_image,
    )

    ground_group_rows = []

    for group_id, start in enumerate(range(0, len(records), group_size)):
        rows = ground_by_image[start:start + group_size]
        end = start + len(rows) - 1

        ratios = [float(x[5]) for x in rows]
        ground_count_sum = sum(int(x[4]) for x in rows)
        roi_pixels_sum = sum(int(x[6]) for x in rows)

        label = f"group_{group_id:04d}_images_{start:06d}_{end:06d}_roi"

        ground_group_rows.append([
            label,
            len(rows),
            rows[0][2],
            rows[-1][2],
            ground_count_sum,
            roi_pixels_sum,
            f"{ground_count_sum / roi_pixels_sum if roi_pixels_sum > 0 else 0.0:.10f}",
            f"{mean(ratios):.10f}",
            f"{std(ratios):.10f}",
            f"{min(ratios) if ratios else 0.0:.10f}",
            f"{max(ratios) if ratios else 0.0:.10f}",
        ])

    write_csv(
        out_dir / f"roi_ground_ratio_by_group_{group_size}.csv",
        [
            "group",
            "n_images",
            "start_file",
            "end_file",
            "ground_count_sum",
            "roi_pixels_sum",
            "pixel_ratio",
            "mean_ground_ratio",
            "std_ground_ratio",
            "min_ground_ratio",
            "max_ground_ratio",
        ],
        ground_group_rows,
    )

    # ------------------------------------------------------------
    # 7. README
    # ------------------------------------------------------------
    readme = out_dir / "README.txt"

    readme.write_text(
        "ROI segmentation statistics output\n"
        "==================================\n\n"
        f"seg_color_dir: {seg_color_dir}\n"
        f"class_visualization_dir: {class_visualization_dir}\n"
        f"out_dir: {out_dir}\n"
        f"num_files: {len(records)}\n"
        f"group_size: {group_size}\n"
        f"ROI ratios: x=[{args.x_start_ratio}, {args.x_end_ratio}], "
        f"y=[{args.y_start_ratio}, {args.y_end_ratio}]\n"
        f"ground_names: {','.join(sorted(ground_names))}\n"
        f"ground_ids: {ground_ids}\n\n"
        "Important:\n"
        "- This script uses *_pred_id.npy, not *_legend.csv alone.\n"
        "- *_legend.csv contains whole-image class ratios only and cannot calculate lower-half statistics by itself.\n"
        "- Default ROI is the lower half of each prediction map.\n\n"
        "Files:\n"
        "- summary_roi_input_files.csv: input pred_id file list and top class within ROI.\n"
        "- roi_class_ratio_by_image_long.csv: per-image, per-class count and ratio within ROI.\n"
        "- roi_class_ratio_by_image_wide.csv: per-image class ratio table within ROI.\n"
        "- roi_class_average_overall.csv: class average within ROI across all images.\n"
        f"- roi_class_average_by_group_{group_size}.csv: class average within ROI for each group of {group_size} images.\n"
        "- roi_ground_ratio_by_image.csv: summed ground candidate class ratio within ROI for each image.\n"
        f"- roi_ground_ratio_by_group_{group_size}.csv: ground ratio within ROI for each group of {group_size} images.\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("===== ROI aggregate done =====")
    print("seg_color_dir:", seg_color_dir)
    print("class_visualization_dir:", class_visualization_dir)
    print("num_files:", len(records))
    print("out_dir:", out_dir)
    print("group_size:", group_size)
    print(
        "roi: y=[%.3f, %.3f], x=[%.3f, %.3f]"
        % (
            args.y_start_ratio,
            args.y_end_ratio,
            args.x_start_ratio,
            args.x_end_ratio,
        )
    )
    print("ground_ids:", ground_ids)
    print("saved:", out_dir / "roi_class_average_overall.csv")
    print("saved:", out_dir / f"roi_class_average_by_group_{group_size}.csv")
    print("saved:", out_dir / "roi_ground_ratio_by_image.csv")
    print("saved:", out_dir / f"roi_ground_ratio_by_group_{group_size}.csv")


if __name__ == "__main__":
    main()
