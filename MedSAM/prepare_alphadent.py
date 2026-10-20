"""
AlphaDent (Sosnin et al. 2025; DSLR intraoral photos, YOLO polygon labels) -> our test layout, for the external test E3.
Inference only: nothing is trained on AlphaDent.

Binary target, built the same way as our dental masks:
  * our dataset:  caries, cavity and crack (Roboflow classes 1-3) -> 1; tooth and everything else -> 0
  * AlphaDent:    the six caries classes (ids 3-8: Caries 1-6 by Black's classification) -> 1;
                  abrasion (0), filling (1), crown (2) and unlabelled pixels -> 0
  * one box per annotated caries instance (tight box around its rasterised polygon), like our one box per lesion;
  * instances smaller than -min_area pixels are dropped from both the boxes and the mask (our rule: > 5 px);
  * overlapping caries instances are merged in the mask (union), as in ours.
Photos without any caries instance cannot be prompted and are skipped (listed in the CSV).

Layout written (same as our test set, so infer_zoom.py / eval_run.py / colab_runs.score() work unchanged):
  <out_root>/alphadent_<split>/MedSAM_test/RGB_AlphaDent/imgs/<name>.npz   imgs (HxWx3 uint8), boxes (N,4) int32 xyxy
  <out_root>/alphadent_<split>/MedSAM_test/RGB_AlphaDent/gts/<name>.npz    gts (HxW uint8, 0/1)
  <out_root>/alphadent_<split>/photos.csv     one row per photo: size, number of caries boxes, classes, skipped or not

Images are read WITHOUT applying EXIF rotation (pixel order as stored), because YOLO polygon coordinates refer to
the stored pixel grid; check_alphadent() in colab_extras compares our masks with AlphaDent's own PNG masks.

Usage:
    python prepare_alphadent.py -src /content/alphadent/<unzipped root> -split valid -out_root /content/work/data
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CARIES = (3, 4, 5, 6, 7, 8)


def find_dir(root, *parts):
    hits = [p for p in Path(root).rglob(parts[-1]) if p.is_dir() and p.parts[-len(parts):] == parts]
    if not hits:
        raise FileNotFoundError(f"No folder .../{'/'.join(parts)} under {root}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def read_polygons(txt, W, H):
    """YOLO segmentation rows: class x1 y1 x2 y2 ... (normalised 0-1). Returns [(class, Nx2 int array)]."""
    out = []
    for line in Path(txt).read_text().split("\n"):
        v = line.split()
        if len(v) < 7:              # class + at least 3 points
            continue
        c = int(float(v[0]))
        xy = np.array(v[1:], dtype=np.float64).reshape(-1, 2)
        pts = np.stack([np.clip(np.round(xy[:, 0] * W), 0, W - 1), np.clip(np.round(xy[:, 1] * H), 0, H - 1)], 1)
        out.append((c, pts.astype(np.int32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", required=True, help="Unzipped AlphaDent root (contains images/<split>/ and labels/<split>/)")
    ap.add_argument("-split", default="valid", choices=["train", "valid"], help="Labelled split to convert (test labels are closed)")
    ap.add_argument("-out_root", required=True)
    ap.add_argument("-n", type=int, default=0, help="Fixed random sample of n photos (seed -seed); 0 = all")
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument("-min_area", type=int, default=6, help="Drop caries instances below this many pixels (ours: > 5 px)")
    ap.add_argument("-caries", default="3,4,5,6,7,8", help="Class ids counted as caries")
    args = ap.parse_args()
    caries = {int(c) for c in args.caries.split(",")}

    img_dir, lab_dir = find_dir(args.src, "images", args.split), find_dir(args.src, "labels", args.split)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.n:
        imgs = sorted(random.Random(args.seed).sample(imgs, min(args.n, len(imgs))))
    ds = f"alphadent_{args.split}"
    out = Path(args.out_root) / ds / "MedSAM_test" / "RGB_AlphaDent"
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    (out / "gts").mkdir(parents=True, exist_ok=True)

    rows = []
    for ip in tqdm(imgs, desc=f"AlphaDent {args.split} -> npz"):
        txt = lab_dir / f"{ip.stem}.txt"
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        row = {"case": f"{ip.stem}.npz", "image": ip.name}
        if img is None or not txt.exists():
            rows.append({**row, "skipped": "unreadable image" if img is None else "no label file"}); continue
        H, W = img.shape[:2]
        polys = read_polygons(txt, W, H)
        gt = np.zeros((H, W), np.uint8)
        boxes, cls, dropped = [], [], 0
        for c, pts in polys:
            if c not in caries:
                continue
            inst = np.zeros((H, W), np.uint8)
            cv2.fillPoly(inst, [pts], 1)
            if int(inst.sum()) < args.min_area:
                dropped += 1; continue
            ys, xs = np.nonzero(inst)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            cls.append(c)
            gt |= inst
        row.update({"height": H, "width": W, "long_side": max(H, W), "caries_boxes": len(boxes),
                    "caries_classes": " ".join(str(c) for c in sorted(set(cls))), "dropped_tiny": dropped,
                    "other_instances": sum(1 for c, _ in polys if c not in caries)})
        if not boxes:
            rows.append({**row, "skipped": "no caries"}); continue
        np.savez_compressed(out / "imgs" / f"{ip.stem}.npz", imgs=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            boxes=np.array(boxes, dtype=np.int32))
        np.savez_compressed(out / "gts" / f"{ip.stem}.npz", gts=gt)
        rows.append({**row, "skipped": ""})

    fields = ["case", "image", "height", "width", "long_side", "caries_boxes", "caries_classes", "dropped_tiny",
              "other_instances", "skipped"]
    with open(Path(args.out_root) / ds / "photos.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    kept = [r for r in rows if not r["skipped"]]
    nb = np.array([r["caries_boxes"] for r in kept]) if kept else np.array([0])
    print(f"{ds}: {len(imgs)} photos read, {len(kept)} written to {out}")
    print(f"  skipped: {len(rows) - len(kept)} ({', '.join(sorted({r['skipped'] for r in rows if r['skipped']})) or '-'})")
    print(f"  caries boxes: total {nb.sum()}, per photo mean {nb.mean():.2f}, max {nb.max()}; "
          f"tiny instances dropped: {sum(r.get('dropped_tiny', 0) or 0 for r in rows)}")
    print(f"  long side: median {int(np.median([r['long_side'] for r in kept]))} px; "
          f"photos above 2,000 px (zoom acts): {sum(r['long_side'] > 2000 for r in kept)} of {len(kept)}")


if __name__ == "__main__":
    main()
