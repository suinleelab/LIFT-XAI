#!/usr/bin/env python3
"""
Plot insertion/deletion curves (mean ± SEM across seeds) per explainer,
with 3 side-by-side subplots (ε_R, ε_DR, ε_IF) for both insertion and deletion.

Inputs (same as before):
  results/{cohort}/insertion_deletion_shuffle_{shuffle}_{learner}_zero_baseline_{zero_baseline}_seed_{i}.pkl

Outputs:
  plots_{cohort}_{learner}_{shuffle|noshuffle}_{zb|nozb}/
    insertion_3wide.png
    deletion_3wide.png
  aggregated_curves.csv
"""

import os
import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

# --- Selection types in the order they'll appear left→right ---
SELECTION_TYPES = ["pseudo_outcome_r", "pseudo_outcome_dr", "if_pehe"]

# Math y-labels for each selection type
YLABELS = {
    "pseudo_outcome_r":  r"$\epsilon_{R}$",
    "pseudo_outcome_dr": r"$\epsilon_{DR}$",
    "if_pehe":           r"$\epsilon_{IF}$",
}

DEFAULT_EXPLAINERS = [
    "saliency",
    "smooth_grad",
    # "gradient_shap",
    # "lime",
    "baseline_lime",
    "baseline_shapley_value_sampling",
    # "marginal_shapley_value_sampling",
    # "integrated_gradients",
    "baseline_integrated_gradients",
    # "kernel_shap"
    # "marginal_shap"
]

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]


# explainer-specific style (label, color, linestyle)
STYLE = {
    "baseline_shapley_value_sampling": ("Shapley value",            PALETTE[0], "-"),
    "marginal_shapley_value_sampling": ("Shapley value",  PALETTE[0], "-"),
    "smooth_grad":                    ("SmoothGrad",                 PALETTE[1], "-"),
    "lime":                           ("Lime",                       PALETTE[2], "-"),
    "baseline_lime":                  ("Lime",                       PALETTE[2], "-"),
    "integrated_gradients":           ("IG",                         PALETTE[3], "-"),
    "baseline_integrated_gradients":  ("IG",                  PALETTE[3], "-"),
    "kernel_shap":                    ("Kernel Shap",                PALETTE[4], "-"),
}
FALLBACK = ("", PALETTE[5], "-")

def coerce_to_dict(x, selection_types=SELECTION_TYPES) -> Dict[str, np.ndarray]:
    """Accept dict or list/tuple aligned to selection_types; return dict of arrays."""
    if isinstance(x, dict):
        return {k: np.asarray(v).reshape(-1) for k, v in x.items()}
    if isinstance(x, (list, tuple)) and len(x) == len(selection_types):
        return {k: np.asarray(v).reshape(-1) for k, v in zip(selection_types, x)}
    raise ValueError(
        "Unexpected insertion/deletion structure. Expected dict keyed by selection_types "
        f"({selection_types}) or list aligned to them."
    )

def aggregate_curves(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate multiple 1D arrays (variable lengths ok) into mean and SEM."""
    max_len = max(a.shape[0] for a in arrays)
    M = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        M[i, :a.shape[0]] = a
    mean = np.nanmean(M, axis=0)
    std = np.nanstd(M, axis=0, ddof=1)
    n = np.sum(~np.isnan(M), axis=0).astype(float)
    sem = np.divide(std, 2*np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 1)
    return mean, sem

def load_rows(files: List[str]):
    """Load all explainer rows from all seed pickles."""
    rows = []
    for fp in files:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        for row in data:  # one per explainer
            rows.append((fp, *row))
    return rows

def chord_values(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    K = y.size
    if K < 2:
        return y
    x = np.arange(1, K + 1)
    return y[0] + (y[-1] - y[0]) * (x - 1) / (K - 1)

def auc_vs_chord(y: np.ndarray, lower_is_better: bool = True, normalize_width: bool = False) -> float:
    """
    Signed area between curve and its chord (chord treated as x-axis).
    If lower_is_better=True, positive means the curve lies below the chord on average.
    If normalize_width=True, divide by (K-1) so values are comparable across lengths.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    K = y.size
    if K < 2:
        return float("nan")

    line = chord_values(y)
    # Signed deviation from chord
    if lower_is_better:
        resid = (line - y)
    else:
        resid = (y - line)

    # Unit-spaced x (1..K). This integrates the residuals.
    x = np.arange(1, K + 1)
    area = float(np.trapz(resid, x))
    return area / (K - 1) if normalize_width else area

def auc_vs_zero_baseline(y: np.ndarray, normalize_width: bool = True) -> float:
    """
    Area under the original curve with 0 as the baseline (classic AUC of y).
    Provided for reference; not using the chord as the x-axis.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    K = y.size
    if K < 2:
        return float("nan")
    x = np.arange(1, K + 1)
    area = float(np.trapz(y, x))
    return area / (K - 1) if normalize_width else area

def _mean_curves_and_aucs(curves_for_kind: Dict[str, List[np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Given curves_for_kind[explainer] -> list of arrays (one per seed),
    return:
      - mean_curves[explainer] -> 1D mean curve
      - aucs[explainer]        -> AUC of mean curve (self-contained, not baseline-aligned)
    """
    mean_curves, aucs = {}, {}
    for explainer, arrs in curves_for_kind.items():
        if not arrs:
            continue
        # aggregate to mean curve
        max_len = max(a.shape[0] for a in arrs)
        M = np.full((len(arrs), max_len), np.nan)
        for i, a in enumerate(arrs):
            M[i, :a.shape[0]] = a
        mean = np.nanmean(M, axis=0)
        # drop trailing NaNs
        valid = np.isfinite(mean)
        mean = mean[valid]
        mean_curves[explainer] = mean
        # auc vs its own x (not baseline-aligned yet)
        if len(mean) >= 2:
            x = np.arange(1, len(mean) + 1)
            aucs[explainer] = float(np.trapz(mean, x))
        else:
            aucs[explainer] = float("nan")
    return mean_curves, aucs



def plot_three_wide(
    curves: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]],
    out_dir: str,
    cohort: str,
    learner: str,
):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    tidy_rows = []

    # ---- Helper: plot one 3-wide figure for the given curve_kind ----
    def _plot_kind(curve_kind: str, fname: str, suptitle: str):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)
        handles, labels = [], []
        auc_rows = []  # rows for CSV/print

        for col, st in enumerate(SELECTION_TYPES):
            ax = axes[col]

            # 1) Plot all mean±SEM curves as you already do, but also keep arrays so we can compute AUCs
            mean_cache = {}  # explainer -> mean curve (for this selection type)
            for explainer, arrs in curves[st][curve_kind].items():
                if not arrs:
                    continue
                label, color, ls = STYLE.get(explainer, (explainer or FALLBACK[0], FALLBACK[1], FALLBACK[2]))
                mean, sem = aggregate_curves(arrs)
                x = np.arange(1, len(mean) + 1)
                line, = ax.plot(x, mean, label=label, color=color, linestyle=ls)
                ax.fill_between(x, mean - sem, mean + sem, alpha=0.2, color=color)

                if label not in labels:
                    handles.append(line)
                    labels.append(label)

                # keep for AUC computation
                mean_cache[explainer] = mean

            ax.set_xlabel("Number of features", size=16)
            ax.set_ylabel(YLABELS.get(st, "PEHE"), size=20)
            is_lower_better = True  # e.g., PEHE-like errors: lower is better

            for explainer, y in mean_cache.items():
                lower_is_better = True if curve_kind == "deletion" else False
                auc_chord_baseline_raw = auc_vs_chord(y, lower_is_better=lower_is_better)
                # (Optional) also report the raw unnormalized area difference
                auc_chord_baseline = auc_vs_chord(y, normalize_width=True)

                auc_rows.append({
                    "curve": curve_kind,                    # "insertion" | "deletion"
                    "selection_type": st,                   # e.g. "pseudo_outcome_r"
                    "explainer": explainer,
                    "AUC_vs_chord_norm": float(auc_chord_baseline),
                    "AUC_vs_chord_raw": float(auc_chord_baseline_raw),
                })

        # Single shared legend below
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=18)
        fig.tight_layout()

        # save figure
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        # 3) Write AUC table for this figure
        auc_df = pd.DataFrame(auc_rows)
        auc_csv = os.path.join(out_dir, fname.replace(".png", "_auc.csv"))
        auc_df.to_csv(auc_csv, index=False)
        saved.append(auc_csv)

        # 4) Also print a compact summary to stdout
        if not auc_df.empty:
            print(f"\nAUC summary for {curve_kind}:")
            cols = ["selection_type", "explainer", "AUC_vs_chord_norm", "AUC_vs_chord_raw"]
            # order by selection_type then AUC desc
            auc_print = (auc_df[cols]
                        .sort_values(["selection_type", "AUC_vs_chord_norm"], ascending=[True, False]))
            print(auc_print.to_string(index=False))

    # Make the two figures
    _plot_kind(curve_kind="insertion", fname="insertion_3wide.png", suptitle="Insertion")
    _plot_kind(curve_kind="deletion",  fname="deletion_3wide.png",  suptitle="Deletion")

    # save tidy CSV
    tidy_df = pd.DataFrame(tidy_rows)
    csv_path = os.path.join(out_dir, "aggregated_curves.csv")
    tidy_df.to_csv(csv_path, index=False)
    saved.append(csv_path)
    return saved

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True, help="e.g., ist3, responder, massive_transfusion")
    p.add_argument("--learner", required=True, help="e.g., XLearner, SLearner, TARNet, etc.")
    p.add_argument("--shuffle", action="store_true", help="Match the training flag (default False)")
    p.add_argument("--zero_baseline", action="store_true", help="Match the training flag (default False)")
    p.add_argument("--results_root", default="results", help="Root directory that holds cohort subfolders")
    p.add_argument("--glob_override", default="", help="Optional custom glob if your filenames differ")
    args = p.parse_args()

    results_dir = os.path.join(args.results_root, args.cohort)
    if args.glob_override:
        pattern = os.path.join(results_dir, args.glob_override)
    else:
        pattern = os.path.join(
            results_dir,
            f"insertion_deletion_shuffle_{args.shuffle}_{args.learner}_"
            f"zero_baseline_{args.zero_baseline}_seed_*.pkl"
        )

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"No files found.\n  Looked for: {pattern}\n"
            "Double-check cohort/learner/shuffle/zero_baseline and file locations."
        )

    rows = load_rows(files)

    # Discover explainers present (we'll still filter by DEFAULT_EXPLAINERS below)
    explainers_found = sorted({r[2] for r in rows})

    # curves[selection_type]["insertion"/"deletion"][explainer] -> list of arrays over seeds
    curves = {
        st: {
            "insertion": {e: [] for e in explainers_found},
            "deletion":  {e: [] for e in explainers_found},
        } for st in SELECTION_TYPES
    }

    # Fill, but restrict to the explainer set you care about
    for (fp, learner_name, explainer_name, insertion_results, deletion_results, *_) in rows:
        if explainer_name not in DEFAULT_EXPLAINERS:
            continue
        ins = coerce_to_dict(insertion_results)
        dele = coerce_to_dict(deletion_results)
        for st in SELECTION_TYPES:
            if st in ins:
                curves[st]["insertion"][explainer_name].append(np.asarray(ins[st]).reshape(-1))
            if st in dele:
                curves[st]["deletion"][explainer_name].append(np.asarray(dele[st]).reshape(-1))

    out_dir = f"plots/{args.cohort}/{args.learner}_{'shuffle' if args.shuffle else 'noshuffle'}_{'zb' if args.zero_baseline else 'nozb'}"

    saved = plot_three_wide(curves, out_dir, args.cohort, args.learner)

    print("Saved:")
    for s in saved:
        print(" -", s)

if __name__ == "__main__":
    main()