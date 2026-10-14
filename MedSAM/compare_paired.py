"""
Paired, per-photo comparison of two runs on the same photos (e.g. zoom vs B0 on validation).

For each photo: diff = metric(candidate) - metric(reference). Reported per group (large / small photos from
the split CSV, and all):
  n, wins, losses, ties, mean and median diff,
  sign test (one-sided, exact, ties dropped): chance of at least this many wins if the two runs were equal,
  Wilcoxon signed-rank test (two-sided, scipy) when available.

Decision rule for the zoom test (step 4), fixed before looking at results:
  a group passes if  sign-test p <= 0.05  AND  mean diff >= +0.03 DSC.

Usage:
  python compare_paired.py -candidate <exp>/zoom_z1/metrics_per_case.csv -reference <exp>/og_ckpt/metrics_per_case.csv \
      -split_csv splits/split_val42_test63.csv [-metric dsc] [-min_gain 0.03] [-alpha 0.05] [-out_csv paired.csv]
"""
import argparse
from math import comb

import numpy as np
import pandas as pd


def sign_test_one_sided(wins, losses):
    n = wins + losses
    if n == 0:
        return float("nan")
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def wilcoxon_p(diff):
    try:
        from scipy.stats import wilcoxon
        d = diff[diff != 0]
        return float(wilcoxon(d).pvalue) if len(d) >= 6 else float("nan")
    except Exception:
        return float("nan")


def summarize(diff, min_gain, alpha, tol=1e-9):
    wins, losses = int((diff > tol).sum()), int((diff < -tol).sum())
    p_sign = sign_test_one_sided(wins, losses)
    mean = float(diff.mean()) if len(diff) else float("nan")
    return {"n": len(diff), "wins": wins, "losses": losses, "ties": len(diff) - wins - losses,
            "mean_diff": round(mean, 4), "median_diff": round(float(np.median(diff)), 4) if len(diff) else float("nan"),
            "sign_p_one_sided": round(p_sign, 4), "wilcoxon_p": round(wilcoxon_p(diff), 4),
            "passes": bool(p_sign <= alpha and mean >= min_gain)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-candidate", required=True)
    ap.add_argument("-reference", required=True)
    ap.add_argument("-split_csv", required=True)
    ap.add_argument("-metric", default="dsc")
    ap.add_argument("-min_gain", type=float, default=0.03)
    ap.add_argument("-alpha", type=float, default=0.05)
    ap.add_argument("-out_csv", default=None, help="Optional: per-photo differences.")
    args = ap.parse_args()

    a = pd.read_csv(args.candidate)[["case", args.metric]].rename(columns={args.metric: "cand"})
    b = pd.read_csv(args.reference)[["case", args.metric]].rename(columns={args.metric: "ref"})
    split = pd.read_csv(args.split_csv).rename(columns={"file": "case"})[["case", "size_group", "box_group"]]
    m = a.merge(b, on="case", how="inner").merge(split, on="case", how="left")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(f"Runs cover different photos: candidate {len(a)}, reference {len(b)}, matched {len(m)}")
    if m.size_group.isna().any():
        raise ValueError("Some photos are missing from the split CSV")
    m["diff"] = m.cand - m.ref

    rows = []
    for name, sel in [("large photos", m.size_group == "large"), ("small photos", m.size_group == "small"),
                      ("all", m.size_group.notna())]:
        r = summarize(m.loc[sel, "diff"].to_numpy(), args.min_gain, args.alpha)
        rows.append({"group": name, **r})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"Metric: {args.metric}   candidate: {args.candidate}\n           reference: {args.reference}")
    print(f"Rule: passes if one-sided sign-test p <= {args.alpha} and mean diff >= {args.min_gain}")
    print(out.to_string(index=False))
    if args.out_csv:
        m.sort_values("diff").to_csv(args.out_csv, index=False)
        print(f"Per-photo differences: {args.out_csv}")


if __name__ == "__main__":
    main()
