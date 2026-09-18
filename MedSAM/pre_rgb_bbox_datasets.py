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
    parser.add_argument("-img_path", type=str, default=None, help="Path to RGB images (single-folder mode).")
    parser.add_argument("-gt_path", type=str, default=None, help="Path to masks (single-folder mode).")
    parser.add_argument("-output_path", type=str, default="data/npz", help="Output root path.")
    parser.add_argument("-modality", type=str, default="RGB", help="Prefix modality.")
    parser.add_argument("-anatomy", type=str, default="Dataset", help="Prefix anatomy/dataset name.")
    parser.add_argument("-img_name_suffix", type=str, default=".jpg", help="Image suffix.")
    parser.add_argument("-gt_name_suffix", type=str, default=".png", help="Mask suffix.")
    parser.add_argument("-bbox_format", type=str, choices=["json", "csv_dir", "mask"], required=True)
    parser.add_argument("-bbox_json", type=str, default=None, help="Path to JSON bbox annotation (single-folder mode).")
    parser.add_argument("-bbox_csv_dir", type=str, default=None, help="Path to dir with per-image csv bbox files (single-folder mode).")
    parser.add_argument("-train_img_path", type=str, default=None, help="Path to train images.")
    parser.add_argument("-train_gt_path", type=str, default=None, help="Path to train masks.")
    parser.add_argument("-train_bbox_format", type=str, choices=["json", "csv_dir", "mask"], default=None)
    parser.add_argument("-train_bbox_json", type=str, default=None, help="Path to train JSON bbox annotation.")
    parser.add_argument("-train_bbox_csv_dir", type=str, default=None, help="Path to train per-image csv bbox dir.")
    parser.add_argument("-test_img_path", type=str, default=None, help="Path to test images.")
    parser.add_argument("-test_gt_path", type=str, default=None, help="Path to test masks.")
    parser.add_argument("-test_bbox_format", type=str, choices=["json", "csv_dir", "mask"], default=None)
    parser.add_argument("-test_bbox_json", type=str, default=None, help="Path to test JSON bbox annotation.")
    parser.add_argument("-test_bbox_csv_dir", type=str, default=None, help="Path to test per-image csv bbox dir.")
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


def resolve_box_sources(bbox_format, bbox_json, bbox_csv_dir):
    bbox_json_map = {}
    bbox_csv = None
    if bbox_format == "json":
        if bbox_json is None:
            raise ValueError("bbox_format=json requires json path")
        bbox_json_map = read_json_boxes(bbox_json)
    elif bbox_format == "csv_dir":
        if bbox_csv_dir is None:
            raise ValueError("bbox_format=csv_dir requires csv dir path")
        bbox_csv = Path(bbox_csv_dir)
    return bbox_json_map, bbox_csv


def preprocess_split(
    split_img_dir,
    split_gt_dir,
    split_name,
    out_root,
    prefix,
    binary_mask,
    bbox_format,
    bbox_json,
    bbox_csv_dir,
    img_name_suffix,
    gt_name_suffix,
):
    split_dir = out_root / split_name / prefix[:-1]
    imgs_out = split_dir / "imgs"
    gts_out = split_dir / "gts"
    imgs_out.mkdir(parents=True, exist_ok=True)
    gts_out.mkdir(parents=True, exist_ok=True)

    bbox_json_map, bbox_csv = resolve_box_sources(bbox_format, bbox_json, bbox_csv_dir)

    mask_files = sorted([p for p in split_gt_dir.iterdir() if p.is_file()])
    valid = []
    for mp in mask_files:
        if not mp.name.endswith(gt_name_suffix):
            continue
        stem = mp.name[: -len(gt_name_suffix)]
        if (split_img_dir / f"{stem}{img_name_suffix}").exists():
            valid.append(stem)

    saved, skipped = 0, 0
    for stem in tqdm(valid, desc=f"Preprocessing {split_name}"):
        img = ensure_rgb_uint8(io.imread(split_img_dir / f"{stem}{img_name_suffix}"))
        gt = ensure_mask(io.imread(split_gt_dir / f"{stem}{gt_name_suffix}"), binary_mask)
        h, w = gt.shape[:2]

        if bbox_format == "json":
            raw_boxes = get_boxes_from_json(stem, bbox_json_map)
        elif bbox_format == "csv_dir":
            raw_boxes = get_boxes_from_csv(stem, bbox_csv)
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

        np.savez_compressed(
            imgs_out / f"{prefix}{stem}.npz",
            imgs=img,
            boxes=np.array(boxes, dtype=np.int32),
        )
        np.savez_compressed(
            gts_out / f"{prefix}{stem}.npz",
            gts=gt,
        )
        saved += 1

    print(f"[{split_name}] saved {saved} cases, skipped {skipped}.")
    print(f"[{split_name}] imgs dir: {imgs_out}")
    print(f"[{split_name}] gts dir: {gts_out}")


def main():
    args = parse_args()
    out_root = Path(args.output_path)
    prefix = f"{args.modality}_{args.anatomy}_"

    explicit_split_mode = all(
        x is not None
        for x in [args.train_img_path, args.train_gt_path, args.test_img_path, args.test_gt_path]
    )

    if explicit_split_mode:
        train_bbox_format = args.train_bbox_format or args.bbox_format
        test_bbox_format = args.test_bbox_format or args.bbox_format
        preprocess_split(
            split_img_dir=Path(args.train_img_path),
            split_gt_dir=Path(args.train_gt_path),
            split_name="MedSAM_train",
            out_root=out_root,
            prefix=prefix,
            binary_mask=args.binary_mask,
            bbox_format=train_bbox_format,
            bbox_json=args.train_bbox_json or args.bbox_json,
            bbox_csv_dir=args.train_bbox_csv_dir or args.bbox_csv_dir,
            img_name_suffix=args.img_name_suffix,
            gt_name_suffix=args.gt_name_suffix,
        )
        preprocess_split(
            split_img_dir=Path(args.test_img_path),
            split_gt_dir=Path(args.test_gt_path),
            split_name="MedSAM_test",
            out_root=out_root,
            prefix=prefix,
            binary_mask=args.binary_mask,
            bbox_format=test_bbox_format,
            bbox_json=args.test_bbox_json or args.bbox_json,
            bbox_csv_dir=args.test_bbox_csv_dir or args.bbox_csv_dir,
            img_name_suffix=args.img_name_suffix,
            gt_name_suffix=args.gt_name_suffix,
        )
        return

    if args.img_path is None or args.gt_path is None:
        raise ValueError("Single-folder mode requires -img_path and -gt_path")

    img_dir = Path(args.img_path)
    gt_dir = Path(args.gt_path)
    bbox_json_map, bbox_csv_dir = resolve_box_sources(args.bbox_format, args.bbox_json, args.bbox_csv_dir)

    train_split_dir = out_root / "MedSAM_train" / prefix[:-1]
    test_split_dir = out_root / "MedSAM_test" / prefix[:-1]
    train_imgs_out = train_split_dir / "imgs"
    train_gts_out = train_split_dir / "gts"
    test_imgs_out = test_split_dir / "imgs"
    test_gts_out = test_split_dir / "gts"
    train_imgs_out.mkdir(parents=True, exist_ok=True)
    train_gts_out.mkdir(parents=True, exist_ok=True)
    test_imgs_out.mkdir(parents=True, exist_ok=True)
    test_gts_out.mkdir(parents=True, exist_ok=True)

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
    for i, stem in enumerate(tqdm(valid, desc="Preprocessing single-folder dataset")):
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

        if i in train_idx:
            imgs_out, gts_out = train_imgs_out, train_gts_out
        else:
            imgs_out, gts_out = test_imgs_out, test_gts_out

        np.savez_compressed(
            imgs_out / f"{prefix}{stem}.npz",
            imgs=img,
            boxes=np.array(boxes, dtype=np.int32),
        )
        np.savez_compressed(
            gts_out / f"{prefix}{stem}.npz",
            gts=gt,
        )
        saved += 1

    print(f"Saved {saved} cases. Skipped {skipped} cases.")
    print(f"Train imgs dir: {train_imgs_out}")
    print(f"Train gts dir: {train_gts_out}")
    print(f"Test imgs dir: {test_imgs_out}")
    print(f"Test gts dir: {test_gts_out}")


if __name__ == "__main__":
    main()
