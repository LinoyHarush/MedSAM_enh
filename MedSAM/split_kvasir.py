"""
Split a Kvasir dataset (Kvasir-SEG or Kvasir-Instrument) into the subfolders listed by its official split files.

For each split file found in -data_dir (train.txt, val.txt, test.txt), the images, masks and boxes of the listed
image ids are copied into <data_dir>/<split>/:
  images/   the photos       (from <data_dir>/images/)
  masks/    the masks        (from <data_dir>/masks/)
  boxes     JSON format: one bounding_boxes.json with the listed ids, taken from the dataset's own *.json file
            (Kvasir-Instrument: bboxes.json)
            CSV format:  bbox/<id>.csv, copied from <data_dir>/bbox/ (Kvasir-SEG)
The box format is detected automatically: a *.json file in -data_dir first, otherwise a bbox/ folder of CSV files.
The output folders are the input of pre_rgb_bbox_datasets.py (-train_* / -test_* arguments).

Our splits: Kvasir-Instrument train.txt / test.txt (472 / 118 images); Kvasir-SEG train.txt / val.txt (880 / 120 images),
whose 120 val images are our Kvasir-SEG test set, so they are written to test/ with -rename val=test.

Usage (from the repo root):
  python MedSAM/split_kvasir.py -data_dir data/kvasir_instruments
  python MedSAM/split_kvasir.py -data_dir data/kvasir_seg -rename val=test

Adapted from the split helper written in notebooks/LiteMedSAM_Baseline_production.ipynb (same splitting logic;
the hard-coded dataset path became the -data_dir argument, and -rename was added).
"""
import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split a Kvasir dataset into train/val/test subfolders by its split files.")
    p.add_argument("-data_dir", required=True,
                   help="Dataset folder with images/, masks/, the split .txt files and the boxes (*.json or bbox/*.csv).")
    p.add_argument("-rename", action="append", default=[], metavar="SPLIT=FOLDER",
                   help="Write a split into a differently named folder, e.g. val=test. Can be repeated.")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.data_dir)
    rename = dict(r.split("=", 1) for r in args.rename)

    # --- 1. Detect the box format
    bb_mode, master_bboxes = None, {}
    bbox_dir = base_dir / "bbox"
    json_candidates = sorted(base_dir.glob("*.json"))
    if json_candidates:
        bb_mode = "json"
        print(f"Detected JSON bounding box format: {json_candidates[0].name}")
        with open(json_candidates[0], "r") as f:
            master_bboxes = json.load(f)
    elif bbox_dir.exists() and list(bbox_dir.glob("*.csv")):
        bb_mode = "csv"
        print(f"Detected CSV bounding box format in folder: {bbox_dir.name}")
    else:
        print("Warning: no bounding box annotations (a *.json file or CSV files in a 'bbox' folder) detected.")

    # --- 2. Process every split that has a .txt file
    for split in ["train", "val", "test"]:
        split_file = base_dir / f"{split}.txt"
        if not split_file.exists():
            continue
        out_dir = base_dir / rename.get(split, split)
        print(f"\nProcessing [{split.upper()}] split -> {out_dir}")

        out_images, out_masks = out_dir / "images", out_dir / "masks"
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
        out_bboxes = None
        if bb_mode == "csv":
            out_bboxes = out_dir / "bbox"
            out_bboxes.mkdir(parents=True, exist_ok=True)

        # image ids, without spaces and file extensions
        with open(split_file, "r") as f:
            file_ids = [line.strip().split(".")[0] for line in f if line.strip()]

        filtered_bboxes = {}
        copied_img = copied_mask = copied_csv = 0
        for file_id in file_ids:
            img_candidates = list((base_dir / "images").glob(f"{file_id}.*"))     # .jpg or .png
            if img_candidates:
                shutil.copy2(img_candidates[0], out_images / img_candidates[0].name)
                copied_img += 1
            mask_candidates = list((base_dir / "masks").glob(f"{file_id}.*"))
            if mask_candidates:
                shutil.copy2(mask_candidates[0], out_masks / mask_candidates[0].name)
                copied_mask += 1
            if bb_mode == "json" and file_id in master_bboxes:
                filtered_bboxes[file_id] = master_bboxes[file_id]
            elif bb_mode == "csv":
                src_csv = bbox_dir / f"{file_id}.csv"
                if src_csv.exists():
                    shutil.copy2(src_csv, out_bboxes / src_csv.name)
                    copied_csv += 1

        n = len(file_ids)
        if bb_mode == "json":
            with open(out_dir / "bounding_boxes.json", "w") as f:
                json.dump(filtered_bboxes, f, indent=2)
            print(f"[{split.upper()}] Done! Images: {copied_img}/{n}, Masks: {copied_mask}/{n}, JSON BBoxes: {len(filtered_bboxes)}/{n}")
        else:
            print(f"[{split.upper()}] Done! Images: {copied_img}/{n}, Masks: {copied_mask}/{n}, CSV BBoxes: {copied_csv}/{n}")

    print("\nDataset split complete.")


if __name__ == "__main__":
    main()
