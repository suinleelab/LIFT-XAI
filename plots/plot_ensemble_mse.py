"""Plotting for ensemble distillation"""
import argparse
import os
import pickle as pkl
import glob

import matplotlib.pyplot as plt
import numpy as np

# Indices in your ensemble pickle structure
METHOD_IDX = 1
TRAIN_MSE_IDX = 3
TEST_MSE_IDX = 5


DEFAULT_EXPLAINERS = [
    "saliency",
    "smooth_grad",
    # "gradient_shap",
    # "lime",
    "baseline_lime",
    "baseline_shapley_value_sampling",
    "marginal_shapley_value_sampling",
    "integrated_gradients",
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
    "marginal_shapley_value_sampling": ("Shapley",                      PALETTE[0], "-"),
    "smooth_grad":                    ("SmoothGrad",                 PALETTE[1], "-"),
    "lime":                           ("Lime",                       PALETTE[2], "-"),
    "baseline_lime":                  ("Lime",                       PALETTE[2], "-"),
    "integrated_gradients":           ("IG",                         PALETTE[3], "-"),
    "baseline_integrated_gradients":  ("IG (baseline)",                        PALETTE[3], "-"),
    "kernel_shap":                    ("Kernel Shap",                PALETTE[4], "-"),
    "loco":                            ("LOCO",                  PALETTE[6], "-"),
    "permucate":                       ("PermuCATE",                  PALETTE[7], "-")
}
FALLBACK = ("", PALETTE[5], "-")

def aggregate_curves(arrays):
    """Aggregate multiple 1D arrays (variable lengths ok) into mean and std."""
    if not arrays:
        return np.array([]), np.array([])

    max_len = max(len(a) for a in arrays)
    M = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        M[i, :len(a)] = 100*a

    mean = np.nanmean(M, axis=0)
    std = np.nanstd(M, axis=0, ddof=1)/10*np.sqrt(len(M))
    return mean, std

def main():
    """Main function"""

    ap = argparse.ArgumentParser(
        description="Plot ensemble-teacher distillation loss (MSE) vs #features from a single ensemble pickle."
    )
    ap.add_argument(
        "--results_root",
        default="results/",
        help="Base results dir (default: results/)",
    )
    ap.add_argument("--dataset", required=True, help="e.g., ist3")
    ap.add_argument("--learner", required=True, help="e.g., RLearner")
    ap.add_argument("--shuffle", action="store_true", help="Match the training flag (default False)")
    ap.add_argument("--zero_baseline", action="store_true", help="Match the training flag (default False)")
    ap.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Explanation methods to plot (e.g., loco marginal_shapley_value_sampling)",
    )
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", default="plots/", help="Output directory for the figure")
    args = ap.parse_args()

    # Multiple ensemble files (with potential seed suffix or wildcards)
    ens_pattern = os.path.join(
        args.results_root,
        args.dataset,
        f"ensemble_shuffle_{args.shuffle}_{args.learner}_zero_baseline_{args.zero_baseline}*.pkl",
    )

    ens_files = sorted(glob.glob(ens_pattern))
    if not ens_files:
        raise SystemExit(f"No ensemble files found matching pattern:\n  {ens_pattern}")

    print(f"Found {len(ens_files)} ensemble files")

    # Dictionary to store arrays for each method across all files
    per_method_all_files = {}

    wanted = set(args.methods) | {f"{m}|ensemble_teacher" for m in args.methods}

    # --- Load methods from all ensemble files ---
    for ens_path in ens_files:
        print(f"Loading: {ens_path}")

        with open(ens_path, "rb") as f:
            entries = pkl.load(f)

        for e in entries:
            if not isinstance(e, (list, tuple)) or len(e) < 6:
                continue
            mname = str(e[METHOD_IDX])
            if mname not in wanted:
                continue

            label = mname.replace("|ensemble_teacher", "")
            vec = e[TRAIN_MSE_IDX if args.split == "train" else TEST_MSE_IDX]
            y = np.asarray(vec, dtype=float).ravel()

            if label not in per_method_all_files:
                per_method_all_files[label] = []
            per_method_all_files[label].append(y)

    if not per_method_all_files:
        raise SystemExit(
            "No matching methods found in the ensemble files. "
            "Check --methods and file contents."
        )

    os.makedirs(args.out, exist_ok=True)

    # Plot
    plt.figure(figsize=(18, 5))

    for mname in sorted(per_method_all_files.keys()):
        print(f"Plotting: {mname} ({len(per_method_all_files[mname])} files)")
        label, color, ls = STYLE.get(mname, (mname or FALLBACK[0], FALLBACK[1], FALLBACK[2]))

        # Aggregate across all files for this method
        arrays = per_method_all_files[mname]
        mean_y, std_y = aggregate_curves(arrays)

        x = np.arange(1, len(mean_y) + 1)  # 1..K features

        # Plot mean line
        alpha = 0.7 if "Shapley" in label else 0.3
        plt.plot(x, mean_y, lw=2, label=label, color=color, linestyle=ls,  alpha=0.5)

        # Add std shading if we have multiple files
        if len(arrays) > 1:
            plt.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2, color=color)

    # plt.title(
    #     f"Ensemble distillation loss vs #features · {args.learner} · {args.split}"
    # )
    plt.xlabel("Number of features", size=18)
    plt.ylabel(r"Distillation loss ($\times 10^{-2}$)", size=18)
    plt.grid(True, alpha=0.3)
    plt.legend(  loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=8, fontsize=18)
    plt.tight_layout()
    out_dir = f"plots/{args.dataset}/{args.learner}_{'shuffle' if args.shuffle else 'noshuffle'}_{'zb' if args.zero_baseline else 'nozb'}"

    out_path = os.path.join(
        out_dir,
        f"ensemble_distillation_mse.png"
    )
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
