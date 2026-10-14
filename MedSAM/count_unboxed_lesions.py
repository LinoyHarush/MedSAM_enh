"""
Count lesions that are in the ground-truth masks but have no box in bboxes.json.

preprocess_tooth_ds.py draws the binary mask, then makes one box per external contour,
keeping only contours with cv2.contourArea > 5. contourArea measures the area enclosed by
the outline, not the pixel count, so a thin crack can have contourArea 0 and get no box,
while its pixels stay in the mask. Photos whose lesions all lose their box are dropped
from the dataset entirely by pre_rgb_bbox_*.py ("skipped ... without valid object boxes").

For every 8-connected lesion in every mask this script checks whether a box was drawn
around it, and reports the ones that were not.

Usage (one split folder = the folder holding masks/ and bboxes.json):
  python count_unboxed_lesions.py -split_dir <.../dental_caries_segmentation_processed/train> -out_csv unboxed_train.csv
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Find lesions in the masks that have no box in bboxes.json.")
    p.add_argument("-split_dir", required=True, help="Folder with masks/*.png and bboxes.json (output of preprocess_tooth_ds.py).")
    p.add_argument("-out_csv", default="unboxed_lesions.csv", help="One row per lesion without a box.")
    p.add_argument("-min_pixels", type=int, default=6, help="Ignore lesions smaller than this many pixels (5 px = the filter's intended noise level).")
    return p.parse_args()


def box_matches(stat, box, tol=1):
    x, y, w, h = stat[:4]
    return (abs(x - box[0]) <= tol and abs(y - box[1]) <= tol
            and abs(x + w - box[2]) <= tol and abs(y + h - box[3]) <= tol)


def inside(stat, box):
    x, y, w, h = stat[:4]
    return x >= box[0] and y >= box[1] and x + w <= box[2] + 1 and y + h <= box[3] + 1


def main():
    args = parse_args()
    split = Path(args.split_dir)
    with open(split / "bboxes.json", "r", encoding="utf-8") as f:
        bboxes = json.load(f)
    masks = sorted((split / "masks").glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"No masks found in {split / 'masks'}")

    rows = []
    s = dict(photos=0, photos_with_lesions=0, lesions=0, lesion_pixels=0, boxed=0,
             unboxed=0, unboxed_pixels=0, unboxed_inside_other_box=0,
             photos_with_unboxed=0, photos_dropped_but_have_lesions=0, tiny_ignored=0)
    per_photo_share = []

    for mp in masks:
        key = mp.stem
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read {mp}")
        gt = (mask > 0).astype(np.uint8)
        s["photos"] += 1
        n, _, stats, _ = cv2.connectedComponentsWithStats(gt, connectivity=8)
        boxes = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in bboxes.get(key, {}).get("bbox", [])]
        if n <= 1:
            continue
        s["photos_with_lesions"] += 1
        photo_px = int(gt.sum())
        photo_unboxed_px = 0
        any_unboxed = False

        for cid in range(1, n):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < args.min_pixels:
                s["tiny_ignored"] += 1
                continue
            s["lesions"] += 1
            s["lesion_pixels"] += area
            if any(box_matches(stats[cid], b) for b in boxes):
                s["boxed"] += 1
                continue
            in_other = any(inside(stats[cid], b) for b in boxes)
            s["unboxed"] += 1
            s["unboxed_pixels"] += area
            s["unboxed_inside_other_box"] += int(in_other)
            photo_unboxed_px += area
            any_unboxed = True
            x, y, w, h = [int(v) for v in stats[cid, :4]]
            rows.append({
                "image_id": key, "img_w": gt.shape[1], "img_h": gt.shape[0],
                "lesion_pixels": area, "bbox_xyxy": [x, y, x + w, y + h],
                "bbox_w": w, "bbox_h": h, "fill_ratio": round(area / max(1, w * h), 3),
                "inside_another_box": in_other, "photo_has_any_box": bool(boxes),
            })

        if any_unboxed:
            s["photos_with_unboxed"] += 1
            if not boxes:
                s["photos_dropped_but_have_lesions"] += 1
            per_photo_share.append(photo_unboxed_px / max(1, photo_px))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "img_w", "img_h", "lesion_pixels", "bbox_xyxy", "bbox_w", "bbox_h",
              "fill_ratio", "inside_another_box", "photo_has_any_box"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Split: {split}")
    print(f"Saved CSV: {out}")
    for k, v in s.items():
        print(f"  {k}: {v}")
    if s["lesions"]:
        print(f"  unboxed lesions: {100 * s['unboxed'] / s['lesions']:.1f}% of lesions, "
              f"{100 * s['unboxed_pixels'] / max(1, s['lesion_pixels']):.2f}% of lesion pixels")
    if per_photo_share:
        a = np.array(per_photo_share)
        print(f"  in photos that have unboxed lesions, share of lesion pixels without a box: "
              f"median {100 * np.median(a):.1f}%, max {100 * a.max():.1f}%")


if __name__ == "__main__":
    main()
