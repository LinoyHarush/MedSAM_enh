import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine separated inference-style RGB npz files into training-style npz files "
            "that contain both imgs and gts in the same .npz per sample."
        )
    )
    parser.add_argument(
        "-img_dir",
        type=str,
        required=True,
        help="Directory containing imgs npz files (each file has imgs + boxes).",
    )
    parser.add_argument(
        "-gt_dir",
        type=str,
        required=True,
        help="Directory containing gts npz files (each file has gts).",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        required=True,
        help="Output directory for training-ready combined npz files.",
    )
    parser.add_argument(
        "-prefix",
        type=str,
        default="",
        help="Optional prefix to add to the output file names.",
    )
    return parser.parse_args()


def read_npz(path: Path):
    try:
        with np.load(path, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}. The source .npz may be incomplete or stored on a flaky filesystem: {e}") from e


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix == ".npz")
    gt_files = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and p.suffix == ".npz"}

    if len(img_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {img_dir}")

    count = 0
    for img_path in tqdm(img_files, desc="Building training npz"):
        stem = img_path.stem
        gt_path = gt_files.get(stem)
        if gt_path is None:
            print(f"Skipping {stem}: no matching GT file found in {gt_dir}")
            continue

        try:
            img_data = read_npz(img_path)
            gt_data = read_npz(gt_path)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            raise

        if "imgs" not in img_data:
            raise KeyError(f"{img_path} is missing 'imgs' key")
        if "gts" not in gt_data:
            raise KeyError(f"{gt_path} is missing 'gts' key")

        obj = {"imgs": img_data["imgs"], "gts": gt_data["gts"]}
        if "boxes" in img_data:
            obj["boxes"] = img_data["boxes"]

        write_path = out_dir / f"{args.prefix}{stem}.npz"
        np.savez_compressed(write_path, **obj)
        count += 1

    print(f"Created {count} training-ready npz files in {out_dir}")


if __name__ == "__main__":
    main()
