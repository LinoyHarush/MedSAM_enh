import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create training-ready RGB npz files for LiteMedSAM. Each output npz contains imgs + gts (and boxes if available)."
    )
    parser.add_argument("-img_path", type=str, required=True, help="Path to RGB images directory.")
    parser.add_argument("-gt_path", type=str, required=True, help="Path to mask directory.")
    parser.add_argument("-output_path", type=str, default="data/MedSAM_train", help="Output directory for combined npz files.")
    parser.add_argument("-modality", type=str, default="RGB", help="Dataset modality prefix.")
    parser.add_argument("-anatomy", type=str, default="Dataset", help="Dataset name prefix.")
    parser.add_argument("-img_name_suffix", type=str, default=".jpg", help="Image extension.")
    parser.add_argument("-gt_name_suffix", type=str, default=".png", help="Mask extension.")
    parser.add_argument("-bbox_format", type=str, choices=["json", "csv_dir", "mask"], required=True)
    parser.add_argument("-bbox_json", type=str, default=None, help="JSON file for all image boxes.")
    parser.add_argument("-bbox_csv_dir", type=str, default=None, help="Folder containing one csv per image.")
    parser.add_argument("--binary_mask", action="store_true", help="Convert masks to 0/1.")
    return parser.parse_args()


def ensure_rgb_uint8(image_data: np.ndarray) -> np.ndarray:
    if image_data.max() > 255:
        image_data = (image_data - image_data.min()) / np.clip(
            image_data.max() - image_data.min(), a_min=1e-8, a_max=None
        )
        image_data = (image_data * 255.0).astype(np.uint8)
    if image_data.ndim == 2:
        image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
    if image_data.ndim == 3 and image_data.shape[-1] > 3:
        image_data = image_data[:, :, :3]
    return image_data.astype(np.uint8)


def ensure_mask(mask_data: np.ndarray, binary_mask: bool) -> np.ndarray:
    if mask_data.ndim == 3:
        mask_data = mask_data[:, :, 0]
    if binary_mask:
        return (mask_data > 0).astype(np.uint8)
    return mask_data.astype(np.uint8)


def read_json_bbox_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_boxes_from_json(stem: str, bbox_map: dict):
    item = bbox_map.get(stem)
    if item is None:
        return []
    return [
        [int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])]
        for b in item.get("bbox", [])
    ]


def get_boxes_from_csv(stem: str, csv_dir: Path):
    csv_path = csv_dir / f"{stem}.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and any(cell.strip() for cell in row):
                rows.append(row)
    if len(rows) <= 1:
        return []
    boxes = []
    start = 1 if "class_name" in rows[0][0].strip().lower() else 0
    for row in rows[start:]:
        if len(row) < 5:
            continue
        boxes.append([int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4]))])
    return boxes


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    return [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]


def clip_box(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(int(xmin), w - 1))
    ymin = max(0, min(int(ymin), h - 1))
    xmax = max(0, min(int(xmax), w - 1))
    ymax = max(0, min(int(ymax), h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def main():
    args = parse_args()
    img_dir = Path(args.img_path)
    gt_dir = Path(args.gt_path)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.modality}_{args.anatomy}_"

    bbox_map = {}
    csv_dir = None
    if args.bbox_format == "json":
        if args.bbox_json is None:
            raise ValueError("-bbox_format json requires -bbox_json")
        bbox_map = read_json_bbox_map(args.bbox_json)
    elif args.bbox_format == "csv_dir":
        if args.bbox_csv_dir is None:
            raise ValueError("-bbox_format csv_dir requires -bbox_csv_dir")
        csv_dir = Path(args.bbox_csv_dir)

    mask_files = sorted(
        p for p in gt_dir.iterdir()
        if p.is_file() and p.name.endswith(args.gt_name_suffix)
    )

    count = 0
    skipped = 0
    for mask_file in tqdm(mask_files, desc="Creating training npz"):
        stem = mask_file.name[: -len(args.gt_name_suffix)]
        img_file = img_dir / f"{stem}{args.img_name_suffix}"
        if not img_file.exists():
            skipped += 1
            continue

        image = ensure_rgb_uint8(io.imread(img_file))
        gt = ensure_mask(io.imread(mask_file), args.binary_mask)
        h, w = gt.shape[:2]

        if args.bbox_format == "json":
            raw_boxes = get_boxes_from_json(stem, bbox_map)
        elif args.bbox_format == "csv_dir":
            raw_boxes = get_boxes_from_csv(stem, csv_dir)
        else:
            raw_boxes = bbox_from_mask(gt)

        boxes = []
        for box in raw_boxes:
            box_clipped = clip_box(box, w, h)
            if box_clipped is not None:
                boxes.append(box_clipped)

        if len(boxes) == 0:
            skipped += 1
            continue

        out_path = out_dir / f"{prefix}{stem}.npz"
        np.savez_compressed(
            out_path,
            imgs=image,
            gts=gt,
            boxes=np.array(boxes, dtype=np.int32),
        )
        count += 1

    print(f"Saved {count} training-ready npz files to {out_dir}")
    print(f"Skipped {skipped} files without valid object boxes or matching image file.")


if __name__ == "__main__":
    main()
