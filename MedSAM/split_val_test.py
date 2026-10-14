"""
Split the 105 lesion test photos into a validation set (42) and a final test set (63).

Why: every decision (zoom or not, best epoch, learning rate) must be made on validation
photos, so the test photos are only used once, for the final comparison.

The split is stratified by
  - photo size: longest side > 2000 px or not (the photos that lose lesions at 256 px), and
  - number of lesion boxes: one or several,
so both parts contain the same mix. It is reproducible (fixed seed) and is saved as a CSV.

Outputs (under -out_root):
  split_val42_test63.csv                                   one row per photo: file, split, size, boxes
  dental_caries_val42/MedSAM_test/RGB_Teeth/{imgs,gts}/    validation photos, same layout as the test set
  dental_caries_test63/MedSAM_test/RGB_Teeth/{imgs,gts}/   final test photos
  dental_caries_val42/train_ready_npz/                     validation as combined npz (imgs, gts, boxes)
                                                           -> the trainer's -val_npz_dir
  Existing prediction folders (segs_<run>/) are copied into both parts as well, so earlier runs
  (B0 = segs_og_ckpt, ...) can be re-scored on each part without running inference again.

The two dataset folders are laid out like data/<dataset>/MedSAM_test/<subset>, so eval_run.py works
on them unchanged:  eval_run.py --dataset dental_caries_val42 ...  /  --dataset dental_caries_test63 ...

Usage:
  python split_val_test.py -test_dir <.../MedSAM_test/RGB_Teeth> -out_root <.../data> [-n_val 42] [-seed 2024]
  python split_val_test.py ... --csv_only     # only write the split CSV
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Stratified validation/test split of the lesion test photos.")
    p.add_argument("-test_dir", required=True, help="Folder with imgs/*.npz (imgs, boxes), gts/*.npz (gts) and optional segs_*/ folders.")
    p.add_argument("-out_root", required=True, help="Where to write the CSV and the two dataset folders (normally the repo's data/).")
    p.add_argument("-n_val", type=int, default=42, help="Number of validation photos (default 42 = 40%% of 105).")
    p.add_argument("-seed", type=int, default=2024)
    p.add_argument("-large_side", type=int, default=2000, help="Photos with a longer side above this count as large.")
    p.add_argument("-subset", default="RGB_Teeth")
    p.add_argument("-val_name", default="dental_caries_val42")
    p.add_argument("-test_name", default="dental_caries_test63")
    p.add_argument("--csv_only", action="store_true", help="Only write the split CSV, copy no files.")
    return p.parse_args()


def describe(test_dir):
    """Photo size and number of boxes per photo. Reads only the masks and boxes, not the photos."""
    rows = []
    for img_npz in sorted((test_dir / "imgs").glob("*.npz")):
        gt_npz = test_dir / "gts" / img_npz.name
        if not gt_npz.exists():
            raise FileNotFoundError(f"No ground truth for {img_npz.name} in {test_dir / 'gts'}")
        with np.load(img_npz, allow_pickle=True) as d:
            n_boxes = int(np.asarray(d["boxes"]).reshape(-1, 4).shape[0])
        with np.load(gt_npz, allow_pickle=True) as d:
            h, w = d["gts"].shape[:2]
        rows.append({"file": img_npz.name, "height": h, "width": w,
                     "long_side": max(h, w), "n_boxes": n_boxes})
    if not rows:
        raise FileNotFoundError(f"No .npz files in {test_dir / 'imgs'}")
    return rows


def stratified_pick(rows, n_val, seed, large_side):
    """Choose n_val photos so each stratum gets its proportional share (largest remainder)."""
    strata = defaultdict(list)
    for r in rows:
        r["size_group"] = "large" if r["long_side"] > large_side else "small"
        r["box_group"] = "multi" if r["n_boxes"] > 1 else "single"
        r["stratum"] = f"{r['size_group']}_{r['box_group']}"
        strata[r["stratum"]].append(r)

    total = len(rows)
    quota = {k: n_val * len(v) / total for k, v in strata.items()}
    take = {k: int(np.floor(q)) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[: n_val - sum(take.values())]:
        take[k] += 1

    rng = np.random.default_rng(seed)
    val = set()
    for k in sorted(strata):
        files = sorted(r["file"] for r in strata[k])
        rng.shuffle(files)
        val.update(files[: take[k]])
    for r in rows:
        r["split"] = "val" if r["file"] in val else "test"
    return strata, take


def copy_part(test_dir, out_dir, files, seg_dirs):
    for sub in ["imgs", "gts"] + [d.name for d in seg_dirs]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(test_dir / "imgs" / f, out_dir / "imgs" / f)
        shutil.copy2(test_dir / "gts" / f, out_dir / "gts" / f)
        for d in seg_dirs:
            if (d / f).exists():
                shutil.copy2(d / f, out_dir / d.name / f)


def write_combined(test_dir, out_dir, files):
    """Validation set for the trainer: one npz per photo with imgs, gts and boxes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        with np.load(test_dir / "imgs" / f, allow_pickle=True) as a, np.load(test_dir / "gts" / f, allow_pickle=True) as b:
            np.savez_compressed(out_dir / f, imgs=a["imgs"], gts=b["gts"], boxes=a["boxes"])


def main():
    args = parse_args()
    test_dir, out_root = Path(args.test_dir), Path(args.out_root)
    rows = describe(test_dir)
    if not 0 < args.n_val < len(rows):
        raise ValueError(f"-n_val must be between 1 and {len(rows) - 1}")
    strata, take = stratified_pick(rows, args.n_val, args.seed, args.large_side)

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"split_val{args.n_val}_test{len(rows) - args.n_val}.csv"
    fields = ["file", "split", "stratum", "size_group", "box_group", "long_side", "height", "width", "n_boxes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["file"])))

    val = sorted(r["file"] for r in rows if r["split"] == "val")
    test = sorted(r["file"] for r in rows if r["split"] == "test")
    print(f"Photos: {len(rows)}  ->  val {len(val)}, test {len(test)}  (seed {args.seed})")
    print(f"{'stratum':<14}{'photos':>8}{'val':>6}{'test':>6}")
    for k in sorted(strata):
        print(f"{k:<14}{len(strata[k]):>8}{take[k]:>6}{len(strata[k]) - take[k]:>6}")
    for part in ("val", "test"):
        sel = [r for r in rows if r["split"] == part]
        print(f"{part:<5} boxes: {sum(r['n_boxes'] for r in sel):>4}   median long side: {int(np.median([r['long_side'] for r in sel]))} px")
    print(f"Saved split CSV: {csv_path}")
    if args.csv_only:
        return

    seg_dirs = sorted(d for d in test_dir.glob("segs_*") if d.is_dir())
    print(f"Copying prediction folders too: {[d.name for d in seg_dirs] or 'none'}")
    val_dir = out_root / args.val_name / "MedSAM_test" / args.subset
    test_out = out_root / args.test_name / "MedSAM_test" / args.subset
    copy_part(test_dir, val_dir, val, seg_dirs)
    copy_part(test_dir, test_out, test, seg_dirs)
    write_combined(test_dir, out_root / args.val_name / "train_ready_npz", val)
    print(f"Validation (eval layout): {val_dir}")
    print(f"Validation (trainer -val_npz_dir): {out_root / args.val_name / 'train_ready_npz'}")
    print(f"Test (eval layout): {test_out}")
    for d in seg_dirs:
        nv = len(list((val_dir / d.name).glob('*.npz')))
        nt = len(list((test_out / d.name).glob('*.npz')))
        if nv + nt != len(rows):
            print(f"  note: {d.name} has predictions for {nv + nt} of {len(rows)} photos")


if __name__ == "__main__":
    main()
