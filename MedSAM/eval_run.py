"""
Inference + evaluation + summary statistics for one trained checkpoint.

Steps:
  1. export_infer_ckpt.py   - convert the checkpoint to the plain state_dict the inference
                               script loads (extracts "model", merges LoRA if present)
  2. CVPR24_LiteMedSAM_infer.py - predict every test image (skipped if all predictions exist)
  3. evaluation/compute_metrics_binary.py - per-case DSC / NSD on binarized masks
  4. summary statistics

Outputs:
  <test_dir>/segs_<run_name>/                              predictions (.npz)
  <test_dir>/metrics_<run_name>_binary.csv                 per-case DSC / NSD
  <exp_root>/<dataset>/<run_name>/metrics_per_case.csv     per-case metrics + n_boxes
  <exp_root>/<dataset>/<run_name>/summary.json             summary statistics
  <exp_root>/<dataset>/runs_summary.csv                    one row per run (rerun replaces its row)

Usage (run from MedSAM/):
  python eval_run.py --run_name og_ckpt --ckpt work_dir/LiteMedSAM/lite_medsam.pth \
      --repo /content/drive/MyDrive/Deep_Learning_Course/MedSAM/MedSAM_enh
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))  # MedSAM/


def run(cmd):
    """Run a helper script from MedSAM/, streaming its output; stop on failure."""
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True, help="Name used for output folders / files")
    ap.add_argument("--ckpt", required=True, help="Checkpoint to evaluate")
    ap.add_argument("--repo", required=True, help="Path to MedSAM_enh (contains data/ and experiments/)")
    ap.add_argument("--dataset", default="dental_caries_segmentation_processed")
    ap.add_argument("--subset", default="RGB_Teeth")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--force_infer", action="store_true", help="Re-run inference even if predictions exist")
    ap.add_argument("--lora_alpha", type=float, default=None,
                    help="Only for LoRA checkpoints without a medsam_lite_lora_*.pth next to them")
    ap.add_argument("--tmp_dir", default="/content/infer_ckpts", help="Where to write the converted checkpoint")
    args = ap.parse_args()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(HERE, args.ckpt)
    if not os.path.isfile(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")

    test_dir = f"{args.repo}/data/{args.dataset}/MedSAM_test/{args.subset}"
    segs_dir = f"{test_dir}/segs_{args.run_name}"
    metrics_csv = f"{test_dir}/metrics_{args.run_name}_binary.csv"
    exp_dir = f"{args.repo}/experiments/{args.dataset}/{args.run_name}"
    runs_csv = f"{args.repo}/experiments/{args.dataset}/runs_summary.csv"
    infer_ckpt = f"{args.tmp_dir}/{args.run_name}.pth"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"===== {args.run_name} =====\ncheckpoint: {ckpt}")

    # 1. Convert checkpoint
    cmd = ["export_infer_ckpt.py", "-ckpt", ckpt, "-out", infer_ckpt]
    if args.lora_alpha is not None:
        cmd += ["--lora_alpha", str(args.lora_alpha)]
    run(cmd)

    # 2. Inference
    n_imgs = len(glob.glob(f"{test_dir}/imgs/*.npz"))
    n_segs = len(glob.glob(f"{segs_dir}/*.npz"))
    if n_imgs == 0:
        sys.exit(f"No test images found in {test_dir}/imgs")
    if n_segs == n_imgs and not args.force_infer:
        print(f"\nSkipping inference: {segs_dir} already has all {n_segs} predictions")
    else:
        run(["CVPR24_LiteMedSAM_infer.py", "-i", f"{test_dir}/imgs/", "-o", segs_dir,
             "-lite_medsam_checkpoint_path", infer_ckpt, "-device", args.device])
        n_segs = len(glob.glob(f"{segs_dir}/*.npz"))
    if n_segs != n_imgs:
        sys.exit(f"Inference incomplete: {n_segs}/{n_imgs} predictions in {segs_dir}")

    # 3. Evaluation (binary metrics)
    run(["evaluation/compute_metrics_binary.py", "-s", segs_dir, "-g", f"{test_dir}/gts", "-csv_dir", metrics_csv])

    # 4. Statistics
    df = pd.read_csv(metrics_csv)
    df["n_boxes"] = [len(np.load(f"{test_dir}/imgs/{c}")["boxes"]) for c in df.case]
    df.to_csv(f"{exp_dir}/metrics_per_case.csv", index=False)

    summary = {"run": args.run_name, "checkpoint": ckpt, "n_cases": len(df)}
    for m in ["dsc", "nsd"]:
        summary.update({
            f"{m}_mean": df[m].mean(), f"{m}_median": df[m].median(), f"{m}_std": df[m].std(),
            f"{m}_single_box_mean": df.loc[df.n_boxes == 1, m].mean(),
            f"{m}_multi_box_mean": df.loc[df.n_boxes > 1, m].mean(),
        })
    summary["n_dsc_zero"] = int((df.dsc == 0).sum())
    summary = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in summary.items()}
    with open(f"{exp_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    runs = pd.read_csv(runs_csv) if os.path.isfile(runs_csv) else pd.DataFrame()
    if len(runs):
        runs = runs[runs["run"] != args.run_name]
    runs = pd.concat([runs, pd.DataFrame([summary])], ignore_index=True)
    runs.to_csv(runs_csv, index=False)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nPer-case metrics: {exp_dir}/metrics_per_case.csv")
    print(f"Summary:          {exp_dir}/summary.json")
    print(f"All runs:         {runs_csv}")


if __name__ == "__main__":
    main()
