import argparse
import csv
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Diagnose box survival after the same resize+pad logic used by train_one_gpu.py. "
            "Supports both training-style npy pairs (imgs/gts) and training-ready npz files."
        )
    )
    p.add_argument("-npz_dir", type=str, default=None, help="Optional directory of training-ready npz files (imgs,gts,boxes).")
    p.add_argument("-data_root", type=str, default=None, help="Optional directory with imgs/ and gts/ subfolders exactly as used in train_one_gpu.py.")
    p.add_argument("-img_dir", type=str, default=None, help="Optional image directory for custom paths.")
    p.add_argument("-gt_dir", type=str, default=None, help="Optional mask directory for custom paths.")
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


def box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_failed_case(img, gt, box_orig, box_256, reason, out_path):
    vis = img.copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)
    if vis.ndim == 2:
        vis = np.repeat(vis[:, :, None], 3, axis=-1)
    else:
        if vis.shape[-1] == 1:
            vis = np.repeat(vis, 3, axis=-1)

    x1, y1, x2, y2 = box_orig
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    mask_overlay = vis.copy()
    mask_bin = (gt > 0).astype(np.uint8)
    mask_overlay[mask_bin > 0] = (255, 0, 0)
    vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

    txt = f"reason={reason} | orig={box_orig} | box256={box_256}"
    cv2.putText(vis, txt[:120], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def iter_npy_pairs(data_root, img_dir, gt_dir):
    if data_root is not None:
        img_dir = Path(data_root) / "imgs"
        gt_dir = Path(data_root) / "gts"
    else:
        if img_dir is None or gt_dir is None:
            return []
        img_dir = Path(img_dir)
        gt_dir = Path(gt_dir)

    if not img_dir.exists() or not gt_dir.exists():
        return []

    pairs = []
    for gt_file in sorted(gt_dir.glob("*.npy")):
        img_file = img_dir / gt_file.name
        if img_file.exists():
            pairs.append((img_file, gt_file))
    return pairs


def iter_npz_files(npz_dir):
    if npz_dir is None:
        return []
    return sorted(Path(npz_dir).glob("*.npz"))


def process_npz_file(npz_path, args, summary, rows, vis_dir):
    try:
        data = np.load(npz_path, allow_pickle=True)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Skipping corrupt npz file: {npz_path} ({exc})")
        summary["corrupt_npz_files"] += 1
        return
    except Exception as exc:
        print(f"Skipping unreadable npz file {npz_path}: {exc}")
        summary["corrupt_npz_files"] += 1
        return

    if not {"imgs", "gts", "boxes"}.issubset(set(data.files)):
        print(f"Skipping malformed npz file {npz_path}: missing imgs/gts/boxes")
        summary["corrupt_npz_files"] += 1
        return

    img = data["imgs"]
    gt = data["gts"]
    boxes = np.asarray(data["boxes"]).reshape(-1, 4)
    if boxes.size == 0:
        return

    gt = np.asarray(gt)
    if gt.ndim == 3:
        gt = gt[..., 0]
    gt_binary = (gt > 0).astype(np.uint8)
    h, w = gt_binary.shape[:2]
    img_256, scale = resize_longest_side(np.asarray(img), target_length=args.image_size, interpolation=cv2.INTER_AREA)
    gt_resized, _ = resize_longest_side(gt_binary, target_length=args.image_size, interpolation=cv2.INTER_NEAREST)
    gt_256 = pad_image(gt_resized, target_size=args.image_size)

    for bi, box in enumerate(boxes):
        summary["total_boxes"] += 1
        box_orig = clip_box_xyxy(box.tolist(), w, h)
        box_256 = transform_box_to_resized(box_orig, scale)
        box_256 = clip_box_xyxy(box_256, args.image_size, args.image_size)

        status = "valid"
        reason = "valid"
        fg_orig = 0
        fg_256 = 0

        if bbox_area(box_256) <= 0:
            status = "invalid"
            reason = "collapsed_after_transform"
            summary["collapsed_after_transform"] += 1
        else:
            x1o, y1o, x2o, y2o = box_orig
            x1n, y1n, x2n, y2n = box_256

            fg_orig = int(np.count_nonzero(gt_binary[y1o:y2o + 1, x1o:x2o + 1] > 0))
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
            "image_id": npz_path.stem,
            "orig_w": w,
            "orig_h": h,
            "box_orig": box_orig,
            "box_256": box_256,
            "orig_box_area": bbox_area(box_orig),
            "box256_area": bbox_area(box_256),
            "fg_pixels_in_orig_box": int(np.count_nonzero(gt_binary[max(0, y1o):min(h, y2o + 1), max(0, x1o):min(w, x2o + 1)] > 0)) if bbox_area(box_orig) > 0 else 0,
            "fg_pixels_in_box256": int(np.count_nonzero(gt_256[max(0, y1n):min(args.image_size, y2n + 1), max(0, x1n):min(args.image_size, x2n + 1)] > 0)) if bbox_area(box_256) > 0 else 0,
            "status": status,
            "reason": reason,
        })

        if args.save_failed_vis and status == "invalid":
            out_name = f"{npz_path.stem}__b{bi}__{reason}.png"
            draw_failed_case(np.asarray(img), gt_binary, box_orig, box_256, reason, vis_dir / out_name)


def process_npy_pair(img_path, gt_path, args, summary, rows, vis_dir):
    img = np.load(img_path, allow_pickle=True)
    gt = np.load(gt_path, allow_pickle=True)

    if gt.ndim == 3:
        gt = gt[..., 0]
    gt = np.asarray(gt, dtype=np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)

    label_ids = np.unique(gt_binary)
    if label_ids.size == 1 and label_ids[0] == 0:
        return

    h, w = gt_binary.shape[:2]
    img_256, scale = resize_longest_side(np.asarray(img), target_length=args.image_size, interpolation=cv2.INTER_AREA)
    gt_resized, _ = resize_longest_side(gt_binary, target_length=args.image_size, interpolation=cv2.INTER_NEAREST)
    gt_256 = pad_image(gt_resized, target_size=args.image_size)

    for label_id in np.unique(gt)[1:]:
        mask = (gt == label_id).astype(np.uint8)
        box_orig = box_from_mask(mask)
        if box_orig is None:
            continue
        summary["total_boxes"] += 1
        box_256 = transform_box_to_resized(box_orig, scale)
        box_256 = clip_box_xyxy(box_256, args.image_size, args.image_size)

        status = "valid"
        reason = "valid"

        if bbox_area(box_256) <= 0:
            status = "invalid"
            reason = "collapsed_after_transform"
            summary["collapsed_after_transform"] += 1
        else:
            x1o, y1o, x2o, y2o = box_orig
            x1n, y1n, x2n, y2n = box_256

            fg_orig = int(np.count_nonzero(mask[y1o:y2o + 1, x1o:x2o + 1] > 0))
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
            "file": img_path.name,
            "box_idx": int(label_id),
            "image_id": img_path.stem,
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
            out_name = f"{img_path.stem}__label_{int(label_id)}__{reason}.png"
            draw_failed_case(np.asarray(img), gt_binary, box_orig, box_256, reason, vis_dir / out_name)


def main():
    args = parse_args()
    if not any([args.npz_dir, args.data_root, args.img_dir, args.gt_dir]):
        raise ValueError("Provide one of: -npz_dir, -data_root, or -img_dir/-gt_dir")

    out_csv = Path(args.out_csv)
    vis_dir = Path(args.vis_dir)
    if args.save_failed_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {
        "total_boxes": 0,
        "valid_boxes": 0,
        "collapsed_after_transform": 0,
        "zero_fg_in_orig_box": 0,
        "zero_fg_in_box256": 0,
        "corrupt_npz_files": 0,
    }

    npz_files = iter_npz_files(args.npz_dir)
    if npz_files:
        for npz_path in tqdm(npz_files, desc="Diagnosing npz files"):
            process_npz_file(npz_path, args, summary, rows, vis_dir)
    else:
        for img_path, gt_path in tqdm(iter_npy_pairs(args.data_root, args.img_dir, args.gt_dir), desc="Diagnosing npy pairs"):
            process_npy_pair(img_path, gt_path, args, summary, rows, vis_dir)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "box_idx", "image_id", "orig_w", "orig_h", "box_orig", "box_256",
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