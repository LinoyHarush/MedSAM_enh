import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose bbox survival after LiteMedSAM train-time resize+pad preprocessing."
    )
    p.add_argument("-npz_dir", type=str, required=True, help="Directory of training-ready npz files (imgs,gts,boxes).")
    p.add_argument("-image_size", type=int, default=256, help="Target square size after pad.")
    p.add_argument("-out_csv", type=str, default="bbox_survival_report.csv", help="Output CSV path.")
    p.add_argument("--save_failed_vis", action="store_true", help="Save visualization for failed boxes.")
    p.add_argument("-vis_dir", type=str, default="bbox_survival_failed_vis", help="Visualization folder.")
    return p.parse_args()


def resize_longest_side(image, target_length=256, interpolation=cv2.INTER_AREA):
    oldh, oldw = image.shape[:2]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
    return cv2.resize(image, (neww, newh), interpolation=interpolation), scale


def pad_image(image, target_size=256):
    h, w = image.shape[:2]
    padh = target_size - h
    padw = target_size - w
    if image.ndim == 3:
        return np.pad(image, ((0, padh), (0, padw), (0, 0)))
    return np.pad(image, ((0, padh), (0, padw)))


def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return [x1, y1, x2, y2]


def transform_box_to_resized(box, scale):
    x1, y1, x2, y2 = box
    return [
        int(x1 * scale + 0.5),
        int(y1 * scale + 0.5),
        int(x2 * scale + 0.5),
        int(y2 * scale + 0.5),
    ]


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def draw_failed_case(img, gt, box_orig, box_256, reason, out_path):
    vis = img.copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)
    if vis.ndim == 2:
        vis = np.repeat(vis[:, :, None], 3, axis=-1)

    x1, y1, x2, y2 = box_orig
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    mask_overlay = vis.copy()
    mask_bin = (gt > 0).astype(np.uint8)
    mask_overlay[mask_bin > 0] = (255, 0, 0)
    vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

    txt = f"reason={reason} | orig={box_orig} | box256={box_256}"
    cv2.putText(vis, txt[:120], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    npz_dir = Path(args.npz_dir)
    out_csv = Path(args.out_csv)
    vis_dir = Path(args.vis_dir)

    if args.save_failed_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    rows = []
    summary = {
        "total_boxes": 0,
        "valid_boxes": 0,
        "collapsed_after_transform": 0,
        "zero_fg_in_orig_box": 0,
        "zero_fg_in_box256": 0,
    }

    for npz_path in tqdm(npz_files, desc="Diagnosing"):
        data = np.load(npz_path, allow_pickle=True)
        img = data["imgs"]
        gt = data["gts"]
        boxes = data["boxes"]

        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = (gt > 0).astype(np.uint8)

        h, w = gt.shape[:2]
        img_256, scale = resize_longest_side(img, target_length=args.image_size, interpolation=cv2.INTER_AREA)
        gt_resized, _ = resize_longest_side(gt, target_length=args.image_size, interpolation=cv2.INTER_NEAREST)

        # pad (same as training flow)
        _ = pad_image(img_256, target_size=args.image_size)
        gt_256 = pad_image(gt_resized, target_size=args.image_size)

        for bi, box in enumerate(boxes):
            summary["total_boxes"] += 1
            box_orig = clip_box_xyxy(box.tolist(), w, h)
            box_256 = transform_box_to_resized(box_orig, scale)
            box_256 = clip_box_xyxy(box_256, args.image_size, args.image_size)

            reason = "valid"
            status = "valid"

            if bbox_area(box_256) <= 0:
                status = "invalid"
                reason = "collapsed_after_transform"
                summary["collapsed_after_transform"] += 1
            else:
                x1o, y1o, x2o, y2o = box_orig
                x1n, y1n, x2n, y2n = box_256

                fg_orig = int(np.count_nonzero(gt[y1o:y2o + 1, x1o:x2o + 1] > 0))
                fg_256 = int(np.count_nonzero(gt_256[y1n:y2n + 1, x1n:x2n + 1] > 0))

                if fg_orig == 0:
                    status = "invalid"
                    reason = "zero_fg_in_orig_box"
                    summary["zero_fg_in_orig_box"] += 1
                elif fg_256 == 0:
                    status = "invalid"
                    reason = "zero_fg_in_box256"
                    summary["zero_fg_in_box256"] += 1
                else:
                    summary["valid_boxes"] += 1

                rows.append({
                    "file": npz_path.name,
                    "box_idx": bi,
                    "orig_w": w,
                    "orig_h": h,
                    "box_orig": box_orig,
                    "box_256": box_256,
                    "orig_box_area": bbox_area(box_orig),
                    "box256_area": bbox_area(box_256),
                    "fg_pixels_in_orig_box": fg_orig,
                    "fg_pixels_in_box256": fg_256,
                    "status": status,
                    "reason": reason,
                })

                if args.save_failed_vis and status == "invalid":
                    out_name = f"{npz_path.stem}__b{bi}__{reason}.png"
                    draw_failed_case(img, gt, box_orig, box_256, reason, vis_dir / out_name)
                continue

            rows.append({
                "file": npz_path.name,
                "box_idx": bi,
                "orig_w": w,
                "orig_h": h,
                "box_orig": box_orig,
                "box_256": box_256,
                "orig_box_area": bbox_area(box_orig),
                "box256_area": bbox_area(box_256),
                "fg_pixels_in_orig_box": -1,
                "fg_pixels_in_box256": -1,
                "status": status,
                "reason": reason,
            })

            if args.save_failed_vis:
                out_name = f"{npz_path.stem}__b{bi}__{reason}.png"
                draw_failed_case(img, gt, box_orig, box_256, reason, vis_dir / out_name)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "box_idx", "orig_w", "orig_h", "box_orig", "box_256",
                "orig_box_area", "box256_area", "fg_pixels_in_orig_box",
                "fg_pixels_in_box256", "status", "reason"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {out_csv}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()