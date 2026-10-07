"""
Plot the DSC / NSD distribution of one metrics CSV (output of compute_metrics*.py)
as box + strip plots, print summary statistics, and save the figure and the statistics.

Outputs (both in --out_dir):
  <out_file>              the figure (e.g. metrics_distribution_after_fix.png)
  <out_file stem>_stats.csv  mean / median / std / min / max / n per metric

Usage:
  python plot_metrics_distribution.py \
      --csv  .../RGB_Teeth/metrics_og_ckpt_binary.csv \
      --out_dir .../experiments/dental_caries_segmentation_processed/og_ckpt \
      --out_file metrics_distribution_after_fix.png \
      --title "LiteMedSAM original checkpoint - Teeth test (binary metrics)"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Metrics CSV with 'dsc' and 'nsd' columns")
    ap.add_argument("--out_dir", required=True, help="Folder to save the figure and stats")
    ap.add_argument("--out_file", required=True, help="Figure file name, e.g. dist.png")
    ap.add_argument("--title", default="", help="Figure title (optional)")
    ap.add_argument("--metrics", nargs="+", default=["dsc", "nsd"], help="Columns to plot")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="Seed for the strip-plot jitter")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [m for m in args.metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not in {args.csv} (has {list(df.columns)})")

    # 1. Statistics
    rows = []
    print(f"--- Inference Statistics: {os.path.basename(args.csv)} (n={len(df)}) ---")
    for m in args.metrics:
        s = df[m]
        row = {"metric": m.upper(), "n": int(s.count()), "mean": s.mean(), "median": s.median(),
               "std": s.std(), "min": s.min(), "max": s.max()}
        rows.append(row)
        print(f"\nMetric: {row['metric']}")
        print(f"  Average (Mean): {row['mean']:.4f}")
        print(f"  Median:         {row['median']:.4f}")
        print(f"  Std Dev:        {row['std']:.4f}")
    stats = pd.DataFrame(rows)

    # 2. Figure (same style as the notebook cell)
    np.random.seed(args.seed)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(7 * len(args.metrics), 6))
    axes = np.atleast_1d(axes)
    palette = sns.color_palette("husl", len(args.metrics))
    for i, (m, row) in enumerate(zip(args.metrics, rows)):
        sns.boxplot(y=df[m], ax=axes[i], color="lightgray", width=0.4, showfliers=False)
        sns.stripplot(y=df[m], ax=axes[i], color=palette[i], alpha=0.6, jitter=True, size=5)
        axes[i].set_title(f"Distribution of {row['metric']}\n"
                          f"mean {row['mean']:.3f} | median {row['median']:.3f} | n={row['n']}",
                          fontsize=14, fontweight="bold")
        axes[i].set_ylabel("Score", fontsize=12)
        axes[i].set_ylim(-0.05, 1.05)
    if args.title:
        fig.suptitle(args.title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 3. Save
    os.makedirs(args.out_dir, exist_ok=True)
    fig_path = os.path.join(args.out_dir, args.out_file)
    stats_path = os.path.join(args.out_dir, os.path.splitext(args.out_file)[0] + "_stats.csv")
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    stats.to_csv(stats_path, index=False, float_format="%.4f")
    print(f"\nSaved figure to: {fig_path}")
    print(f"Saved stats to:  {stats_path}")


if __name__ == "__main__":
    main()
