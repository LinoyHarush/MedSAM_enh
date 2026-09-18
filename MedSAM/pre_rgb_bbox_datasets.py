import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess 2D RGB datasets with bbox annotations (JSON or per-image CSV) into MedSAM npz."
    )
    parser.add_argument("-img_path", type=str, required=True, help="Path to RGB images.")
    parser.add_argument("-gt_path", type=str, required=True, help="Path to masks.")
    parser.add_argument("-output_path", type=str, default="data/npz", help="Output root path.")
    parser.add_argument("-modality", type=str, default="RGB", help="Prefix modality.")
    parser.add_argument("-anatomy", type=str, default="Dataset", help="Prefix anatomy/dataset name.")
    parser.add_argument("-img_name_suffix", type=str, default=".jpg", help="Image suffix.")
    parser.add_argument("-gt_name_suffix", type=str, default=".png", help="Mask suffix.")
    parser.add_argument("-bbox_format", type=str, choices=["json", "csv_dir", "mask"], required=True)
    parser.add_argument("-bbox_json", type=str, default=None, help="Path to JSON bbox annotation.")
    parser.add_argument("-bbox_csv_dir", type=str, default=None, help="Path to dir with per-image csv bbox files.")
    parser.add_argument("-train_ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("-seed", type=int, default=2024, help="Split seed.")
    parser.add_argument("--binary_mask", action="store_true", help="Convert mask to 0/1.")
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


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def read_json_boxes(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_boxes_from_json(stem: str, bbox_json_map: dict):
    item = bbox_json_map.get(stem)
    if item is None:
        return []
    boxes = []
    for b in item.get("bbox", []):
        boxes.append([int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])])
    return boxes


def get_boxes_from_csv(stem: str, bbox_csv_dir: Path):
    csv_path = bbox_csv_dir / f"{stem}.csv"
    if not csv_path.exists():
        return []
    boxes = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if len(r) > 0 and any(c.strip() for c in r)]
    if len(rows) == 0:
        return boxes
    start_idx = 1 if "class_name" in rows[0][0].strip().lower() else 0
    for row in rows[start_idx:]:
        if len(row) < 5:
            continue
        boxes.append([int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4]))])
    return boxes


def clip_box(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(int(xmin), w - 1))
    xmax = max(0, min(int(xmax), w - 1))
    ymin = max(0, min(int(ymin), h - 1))
    ymax = max(0, min(int(ymax), h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def main():
    args = parse_args()
    img_dir = Path(args.img_path)
    gt_dir = Path(args.gt_path)
    out_root = Path(args.output_path)
    prefix = f"{args.modality}_{args.anatomy}_"

    train_dir = out_root / "MedSAM_train" / prefix[:-1]
    test_dir = out_root / "MedSAM_test" / prefix[:-1]
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    bbox_json_map = {}
    bbox_csv_dir = None
    if args.bbox_format == "json":
        if args.bbox_json is None:
            raise ValueError("bbox_format=json requires -bbox_json")
        bbox_json_map = read_json_boxes(args.bbox_json)
    elif args.bbox_format == "csv_dir":
        if args.bbox_csv_dir is None:
            raise ValueError("bbox_format=csv_dir requires -bbox_csv_dir")
        bbox_csv_dir = Path(args.bbox_csv_dir)

    mask_files = sorted([p for p in gt_dir.iterdir() if p.is_file() and p.name.endswith(args.gt_name_suffix)])
    valid = []
    for mp in mask_files:
        stem = mp.name[: -len(args.gt_name_suffix)]
        if (img_dir / f"{stem}{args.img_name_suffix}").exists():
            valid.append(stem)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(valid))
    rng.shuffle(idx)
    split_idx = int(len(valid) * args.train_ratio)
    train_idx = set(idx[:split_idx].tolist())

    saved, skipped = 0, 0
    for i, stem in enumerate(tqdm(valid, desc="Preprocessing RGB dataset")):
        img = ensure_rgb_uint8(io.imread(img_dir / f"{stem}{args.img_name_suffix}"))
        gt = ensure_mask(io.imread(gt_dir / f"{stem}{args.gt_name_suffix}"), args.binary_mask)
        h, w = gt.shape[:2]

        if args.bbox_format == "json":
            raw_boxes = get_boxes_from_json(stem, bbox_json_map)
        elif args.bbox_format == "csv_dir":
            raw_boxes = get_boxes_from_csv(stem, bbox_csv_dir)
        else:
            bm = bbox_from_mask(gt)
            raw_boxes = [bm] if bm is not None else []

        boxes = []
        for b in raw_boxes:
            cb = clip_box(b, w=w, h=h)
            if cb is not None:
                boxes.append(cb)
        if len(boxes) == 0:
            skipped += 1
            continue

        save_dir = train_dir if i in train_idx else test_dir
        np.savez_compressed(
            save_dir / f"{prefix}{stem}.npz",
            imgs=img,
            gts=gt,
            boxes=np.array(boxes, dtype=np.int32),
        )
        saved += 1

    print(f"Saved {saved} npz files.")
    print(f"Skipped {skipped} files (missing/invalid boxes).")
    print(f"Train npz dir: {train_dir}")
    print(f"Test npz dir: {test_dir}")


if __name__ == "__main__":
    main()
