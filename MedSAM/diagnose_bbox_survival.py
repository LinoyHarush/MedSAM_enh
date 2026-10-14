import argparse
import csv
import errno
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Diagnose box survival after the same resize+pad logic used by the trainers. "
            "Supports training-ready npz files (imgs,gts,boxes; one row per stored box) and "
            "training-style npy pairs (imgs/gts; one row per label id, as train_one_gpu.py samples them)."
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


def to_256(mask, image_size):
    """Nearest-neighbour resize (longest side) + pad, exactly as the trainers do for masks."""
    resized, _ = resize_longest_side(mask, target_length=image_size, interpolation=cv2.INTER_NEAREST)
    return pad_image(resized, target_size=image_size)


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


def count_in_box(mask, box):
    """Foreground pixels inside an inclusive xyxy box (works for zero-width/height boxes too)."""
    x1, y1, x2, y2 = box
    return int(np.count_nonzero(mask[y1:y2 + 1, x1:x2 + 1] > 0))


def box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def lesion_component(labels, stats, box):
    """
    Return the connected-component id that a stored box was drawn around, or 0 if none.

    preprocess_tooth_ds.py draws one box per external contour (cv2.boundingRect), i.e. per
    8-connected component, with xmax = x + w. So the matching component has the same
    top-left corner and size. Falls back to the component with the most pixels inside the box.
    """
    x1, y1, x2, y2 = box
    best, best_px = 0, 0
    for cid in range(1, stats.shape[0]):
        cx, cy, cw, ch, area = stats[cid]
        if abs(cx - x1) <= 1 and abs(cy - y1) <= 1 and abs(cx + cw - x2) <= 1 and abs(cy + ch - y2) <= 1:
            return cid
        # cheap overlap test before counting pixels
        if cx > x2 or cy > y2 or cx + cw - 1 < x1 or cy + ch - 1 < y1:
            continue
        px = int(np.count_nonzero(labels[y1:y2 + 1, x1:x2 + 1] == cid))
        if px > best_px:
            best, best_px = cid, px
    return best


def draw_failed_case(img, gt, box_orig, box_256, reason, out_path):
    vis = np.asarray(img).copy()
    if vis.dtype != np.uint8:
        if vis.max() <= 1.0:  # npy images are normalised to [0, 1]
            vis = vis * 255.0
        vis = np.clip(vis, 0, 255).astype(np.uint8)
    if vis.ndim == 2:
        vis = np.repeat(vis[:, :, None], 3, axis=-1)
    elif vis.shape[-1] == 1:
        vis = np.repeat(vis, 3, axis=-1)

    x1, y1, x2, y2 = box_orig
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    mask_overlay = vis.copy()
    mask_overlay[gt > 0] = (255, 0, 0)
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


def is_storage_error(exc):
    """True for errors that mean the disk/Drive mount is gone, not that the file is bad."""
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in (
        errno.ENOTCONN, errno.EIO, errno.ESTALE, errno.ETIMEDOUT
    )


def classify(box_256, own_256, any_256, fg_orig):
    """Shared status logic. Returns (status, reason)."""
    if fg_orig == 0:
        return "invalid", "zero_fg_in_orig_box"
    if bbox_area(box_256) <= 0:
        # zero width or height at 256 px; the multibox trainer replaces the target with the box rectangle
        return "invalid", "collapsed_after_transform"
    if own_256 == 0 and any_256 > 0:
        # this lesion is gone at 256 px, but a neighbouring lesion's pixels are inside its box
        return "invalid", "lesion_vanished_neighbor_in_box"
    if own_256 == 0:
        return "invalid", "zero_fg_in_box256"
    return "valid", "valid"


def process_npz_file(npz_path, args, summary, rows, vis_dir):
    try:
        data = np.load(npz_path, allow_pickle=True)
        files = set(data.files)
        if not {"imgs", "gts", "boxes"}.issubset(files):
            print(f"Skipping malformed npz file {npz_path}: missing imgs/gts/boxes")
            summary["corrupt_npz_files"] += 1
            return
        gt = np.asarray(data["gts"])
        boxes = np.asarray(data["boxes"]).reshape(-1, 4)
        img = np.asarray(data["imgs"]) if args.save_failed_vis else None  # the image is only needed for pictures
    except Exception as exc:
        if is_storage_error(exc) or isinstance(exc.__context__, OSError) and is_storage_error(exc.__context__):
            raise RuntimeError(
                f"Lost access to the data while reading {npz_path} ({exc}). "
                "The Drive mount probably disconnected: copy the folder to local disk and re-run."
            ) from exc
        print(f"Skipping corrupt npz file: {npz_path} ({exc})")
        summary["corrupt_npz_files"] += 1
        return

    if boxes.size == 0:
        return
    summary["files"] += 1

    if gt.ndim == 3:
        gt = gt[..., 0]
    gt_binary = (gt > 0).astype(np.uint8)
    h, w = gt_binary.shape[:2]
    scale = args.image_size * 1.0 / max(h, w)
    gt_256 = to_256(gt_binary, args.image_size)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(gt_binary, connectivity=8)

    for bi, box in enumerate(boxes):
        summary["total_boxes"] += 1
        box_orig = clip_box_xyxy(box.tolist(), w, h)
        box_256 = clip_box_xyxy(transform_box_to_resized(box_orig, scale), args.image_size, args.image_size)

        fg_orig = count_in_box(gt_binary, box_orig)
        any_256 = count_in_box(gt_256, box_256)

        cid = lesion_component(labels, stats, box_orig)
        if cid > 0:
            own = (labels == cid).astype(np.uint8)
            lesion_px_orig = int(stats[cid, cv2.CC_STAT_AREA])
            own_256 = int(np.count_nonzero(to_256(own, args.image_size)))
        else:
            lesion_px_orig, own_256 = 0, 0
            summary["box_without_matching_lesion"] += 1

        status, reason = classify(box_256, own_256, any_256, fg_orig)
        summary["valid_boxes" if status == "valid" else reason] += 1

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
            "fg_pixels_in_orig_box": fg_orig,
            "fg_pixels_in_box256": any_256,
            "lesion_pixels_orig": lesion_px_orig,
            "lesion_pixels_256": own_256,
            "status": status,
            "reason": reason,
        })

        if args.save_failed_vis and status == "invalid":
            out_name = f"{npz_path.stem}__b{bi}__{reason}.png"
            draw_failed_case(img, gt_binary, box_orig, box_256, reason, vis_dir / out_name)


def process_npy_pair(img_path, gt_path, args, summary, rows, vis_dir):
    gt = np.load(gt_path, allow_pickle=True)
    if gt.ndim == 3:
        gt = gt[..., 0]
    gt = np.asarray(gt, dtype=np.uint8)
    label_ids = np.unique(gt)[1:]
    if label_ids.size == 0:
        return
    summary["files"] += 1
    img = np.load(img_path, allow_pickle=True) if args.save_failed_vis else None

    h, w = gt.shape[:2]
    scale = args.image_size * 1.0 / max(h, w)
    gt_256 = to_256(gt, args.image_size)

    for label_id in label_ids:
        mask = (gt == label_id).astype(np.uint8)
        box_orig = box_from_mask(mask)
        summary["total_boxes"] += 1
        box_256 = clip_box_xyxy(transform_box_to_resized(box_orig, scale), args.image_size, args.image_size)

        fg_orig = int(np.count_nonzero(mask))
        own_256 = int(np.count_nonzero(gt_256 == label_id))  # what train_one_gpu.py samples from
        status, reason = classify(box_256, own_256, own_256, fg_orig)
        summary["valid_boxes" if status == "valid" else reason] += 1

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
            "fg_pixels_in_box256": count_in_box(gt_256, box_256),
            "lesion_pixels_orig": fg_orig,
            "lesion_pixels_256": own_256,
            "status": status,
            "reason": reason,
        })

        if args.save_failed_vis and status == "invalid":
            out_name = f"{img_path.stem}__label_{int(label_id)}__{reason}.png"
            draw_failed_case(img, (gt > 0).astype(np.uint8), box_orig, box_256, reason, vis_dir / out_name)


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
        "files": 0,
        "total_boxes": 0,
        "valid_boxes": 0,
        "collapsed_after_transform": 0,
        "zero_fg_in_orig_box": 0,
        "zero_fg_in_box256": 0,
        "lesion_vanished_neighbor_in_box": 0,
        "box_without_matching_lesion": 0,
        "corrupt_npz_files": 0,
    }

    if args.npz_dir:
        npz_files = iter_npz_files(args.npz_dir)
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {args.npz_dir}")
        for npz_path in tqdm(npz_files, desc="Diagnosing npz files"):
            process_npz_file(npz_path, args, summary, rows, vis_dir)
    else:
        pairs = iter_npy_pairs(args.data_root, args.img_dir, args.gt_dir)
        if not pairs:
            raise FileNotFoundError("No imgs/gts .npy pairs found")
        for img_path, gt_path in tqdm(pairs, desc="Diagnosing npy pairs"):
            process_npy_pair(img_path, gt_path, args, summary, rows, vis_dir)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "box_idx", "image_id", "orig_w", "orig_h", "box_orig", "box_256",
                "orig_box_area", "box256_area", "fg_pixels_in_orig_box",
                "fg_pixels_in_box256", "lesion_pixels_orig", "lesion_pixels_256", "status", "reason"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {out_csv}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if rows:
        px = np.array([r["lesion_pixels_256"] for r in rows])
        valid = px[px > 0]
        print("Lesion size at 256 px (surviving lesions):")
        print(f"  lesions with 1-4 px: {int(((px > 0) & (px < 5)).sum())}, 5-19 px: {int(((px >= 5) & (px < 20)).sum())}, "
              f"20+ px: {int((px >= 20).sum())}")
        if valid.size:
            print(f"  median {int(np.median(valid))} px, 10th percentile {int(np.percentile(valid, 10))} px")


if __name__ == "__main__":
    main()
