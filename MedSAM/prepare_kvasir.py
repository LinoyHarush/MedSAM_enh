"""
Kvasir-SEG (1,000 colonoscopy polyp images with masks) -> our test layout, for the forgetting check:
does training on dental photos hurt LiteMedSAM on a different medical task?

Writes <out_root>/kvasir_seg/MedSAM_test/RGB_Teeth/{imgs,gts}/<name>.npz in exactly the layout of our dental test set
(imgs: HxWx3 uint8 + boxes (N,4) int32 xyxy; gts: HxW uint8 0/1), so infer_zoom.py, eval_run.py and colab_runs.score()
work unchanged. The folder name RGB_Teeth is only the layout name our scripts expect.

Boxes are made the same way as our dental boxes: one tight box per connected polyp region of the mask (8-connected),
ignoring specks below -min_area pixels. Kvasir's own kavsir_bboxes.json is not used, so both datasets follow one rule.

Usage (Colab):
    python prepare_kvasir.py -src /content/kvasir/Kvasir-SEG -out_root /content/work/data [-n 0] [-seed 0]
-src is the unzipped folder that contains images/ and masks/ (searched recursively if nested).
-n > 0 takes a fixed random sample of n images (seed -seed); 0 = all 1,000.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def find_dir(root, name):
    hits = [p for p in Path(root).rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"No '{name}' folder under {root}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def boxes_from_mask(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append([x, y, x + w - 1, y + h - 1])
    return np.array(boxes, dtype=np.int32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", required=True, help="Unzipped Kvasir-SEG folder (contains images/ and masks/).")
    ap.add_argument("-out_root", required=True, help="Writes <out_root>/kvasir_seg/MedSAM_test/RGB_Teeth/")
    ap.add_argument("-n", type=int, default=0, help="Fixed random sample of n images; 0 = all.")
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument("-min_area", type=int, default=100, help="Ignore mask regions smaller than this (px).")
    args = ap.parse_args()

    img_dir, mask_dir = find_dir(args.src, "images"), find_dir(args.src, "masks")
    names = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if args.n:
        names = sorted(random.Random(args.seed).sample(names, args.n))
    out = Path(args.out_root) / "kvasir_seg" / "MedSAM_test" / "RGB_Teeth"
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    (out / "gts").mkdir(parents=True, exist_ok=True)

    n_boxes, skipped = [], []
    for nm in tqdm(names, desc="Kvasir-SEG -> npz"):
        img = cv2.imread(str(img_dir / f"{nm}.jpg"), cv2.IMREAD_COLOR)
        mp = next((mask_dir / f"{nm}{e}" for e in (".jpg", ".png") if (mask_dir / f"{nm}{e}").exists()), None)
        if img is None or mp is None:
            skipped.append(nm); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)   # JPEG masks: threshold
        boxes = boxes_from_mask(mask, args.min_area)
        if len(boxes) == 0:
            skipped.append(nm); continue
        # ground truth = only the regions that got a box (specks below min_area are dropped from both)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= args.min_area]
        gt = np.isin(labels, keep).astype(np.uint8)
        np.savez_compressed(out / "imgs" / f"{nm}.npz", imgs=img, boxes=boxes)
        np.savez_compressed(out / "gts" / f"{nm}.npz", gts=gt)
        n_boxes.append(len(boxes))

    nb = np.array(n_boxes)
    print(f"Wrote {len(nb)} images to {out}  (skipped {len(skipped)}: {skipped[:5]})")
    print(f"Boxes per image: mean {nb.mean():.2f}, max {nb.max()}, images with >1 polyp: {(nb > 1).sum()}")


if __name__ == "__main__":
    main()
