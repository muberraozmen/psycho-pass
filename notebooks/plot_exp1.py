"""
Experiment 1 importance figure.

Generates `paper/plots/exp1_importance.pdf` — a 2x2 panel of horizontal
bar charts showing permutation importance for each (encoder, turn-count)
sub-experiment, with LR and GB shown side-by-side per factor.

Story: when n_turns is in the feature set (top row), it dominates
classification; when removed (bottom row), the path-length proxy
(l2norm distance) absorbs that role — same pattern under both encoders.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Pretty names — map raw factor IDs to LaTeX-style symbols used in the paper.
# Keep these consistent with main.tex (\Lpath, \Vel, \meanspeed, \StretchDec, ...).
# ---------------------------------------------------------------------------
PRETTY = {
    "executed_turns":                                              r"$T_{\operatorname{exec}}$",
    "l2norm_semantic_conversation_distance":                       r"$L$",
    "l2norm_semantic_conversation_speed":                          r"$\bar{s}$",
    "l2norm_semantic_conversation_speed_std":                      r"$\sigma_s$",
    "l2norm_semantic_conversation_velocity":                       r"$V$",
    "l2norm_semantic_conversation_displacement":                   r"$D$",
    "catch22_semantic_conversation_stretch_decreasing_min":        r"$\mathrm{SD}_{\min}$",
    "catch22_semantic_conversation_stretch_high_mean":             r"$\mathrm{SH}_{\mathrm{mean}}$",
    "catch22_semantic_conversation_outlier_timing_pos_mean":       r"$\mathrm{OT}^{+}_{\mathrm{mean}}$",
    "catch22_semantic_conversation_outlier_timing_neg_min":        r"$\mathrm{OT}^{-}_{\min}$",
    "catch22_semantic_conversation_outlier_timing_neg_std":        r"$\mathrm{OT}^{-}_{\mathrm{std}}$",
    "catch22_semantic_conversation_high_fluctuation_min":          r"$\mathrm{HF}_{\min}$",
    "l2norm_lexical_conversation_distance":                        r"$L$",
    "l2norm_lexical_conversation_speed":                           r"$\bar{s}$",
    "l2norm_lexical_conversation_velocity":                        r"$V$",
}

# ---------------------------------------------------------------------------
# Factor importances per (sub-experiment, classifier).  Numbers transcribed
# from experiments/analysis/experiment1{a,b,c,d}/out.log.
# Each entry: (factor_id, importance, std).  Order = ranked by importance.
# ---------------------------------------------------------------------------
DATA = {
    ("1a", "LR"): [
        ("executed_turns",                                              0.267102, 0.007894),
        ("l2norm_semantic_conversation_distance",                       0.109332, 0.003354),
        # ("catch22_semantic_conversation_stretch_decreasing_min",        0.009459, 0.001760),
        # ("catch22_semantic_conversation_stretch_high_mean",             0.001540, 0.000729),
    ],
    ("1a", "GB"): [
        ("executed_turns",                                              0.299541, 0.007255),
        ("l2norm_semantic_conversation_distance",                       0.000953, 0.000476),
    ],
    ("1b", "LR"): [
        ("l2norm_semantic_conversation_distance",                       0.330904, 0.005909),
        ("l2norm_semantic_conversation_speed",                          0.037226, 0.003784),
        # ("catch22_semantic_conversation_stretch_decreasing_min",        0.009702, 0.001696),
        # ("catch22_semantic_conversation_stretch_high_mean",             0.004059, 0.001348),
        # ("catch22_semantic_conversation_outlier_timing_pos_mean",       0.000383, 0.000358),
    ],
    ("1b", "GB"): [
        ("l2norm_semantic_conversation_distance",                       0.121540, 0.003162),
        ("l2norm_semantic_conversation_velocity",                       0.103630, 0.002973),
        ("l2norm_semantic_conversation_speed",                          0.016989, 0.001415),
        # ("l2norm_semantic_conversation_displacement",                   0.000672, 0.000320),
        # ("catch22_semantic_conversation_outlier_timing_neg_min",        0.000632, 0.000223),
        # ("catch22_semantic_conversation_high_fluctuation_min",          0.000486, 0.000000),
        # ("catch22_semantic_conversation_outlier_timing_pos_mean",       0.000340, 0.000223),
        # ("l2norm_semantic_conversation_speed_std",                      0.000292, 0.000238),
        # ("catch22_semantic_conversation_outlier_timing_neg_std",        0.000292, 0.000238),
    ],
    ("1c", "LR"): [
        ("executed_turns",                                              0.125772, 0.003893),
        ("l2norm_lexical_conversation_distance",                        0.116428, 0.003157),
        # ("l2norm_lexical_conversation_speed",                           0.001223, 0.000820),
    ],
    ("1c", "GB"): [
        ("executed_turns",                                              0.299856, 0.007234),
    ],
    ("1d", "LR"): [
        ("l2norm_lexical_conversation_distance",                        0.260240, 0.006875),
        ("l2norm_lexical_conversation_speed",                           0.039271, 0.002256),
    ],
    ("1d", "GB"): [
        ("l2norm_lexical_conversation_velocity",                        0.125334, 0.002616),
        ("l2norm_lexical_conversation_distance",                        0.115819, 0.002608),
        # ("l2norm_lexical_conversation_speed",                           0.000907, 0.000259),
    ],
}

PANEL_ORDER = [
    ("1a", "Semantic, with $T_{\operatorname{exec}}$"),
    ("1b", "Semantic, without $T_{\operatorname{exec}}$"),
    ("1c", "Lexical, with $T_{\operatorname{exec}}$"),
    ("1d", "Lexical, without $T_{\operatorname{exec}}$"),
]

# ---------------------------------------------------------------------------
# Plot styling knobs: set these once here.
# ---------------------------------------------------------------------------
SEABORN_THEME = "white"  # e.g., "whitegrid", "darkgrid", "white", "dark", "ticks"
PAPER_COLORS = {"LR": sns.color_palette("pastel")[1], "GB": sns.color_palette("pastel")[4]}
FONT_SIZE = 18


def panel_factor_order(exp_id):
    """Return ordered (deduplicated) factor list for a panel, ranked by max
    importance across the two classifiers (descending)."""
    pooled = {}
    for clf in ("LR", "GB"):
        for fid, imp, _ in DATA[(exp_id, clf)]:
            pooled[fid] = max(pooled.get(fid, 0.0), imp)
    return sorted(pooled, key=pooled.get, reverse=True)


def lookup(exp_id, clf, fid):
    """Return (importance, std) for a (panel, classifier, factor); zeros if absent."""
    for f, imp, std in DATA[(exp_id, clf)]:
        if f == fid:
            return imp, std
    return 0.0, 0.0


def main():
    sns.set_theme(
        style=SEABORN_THEME,
        context="paper",
        rc={
            "font.family": "serif",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    bar_h = 0.36

    for ax, (exp_id, title) in zip(axes.flat, PANEL_ORDER):
        factors = panel_factor_order(exp_id)
        # Display top to bottom in descending importance: invert so top is largest.
        ypos = np.arange(len(factors))[::-1]

        for offset, clf, color in [(+bar_h / 2, "LR", PAPER_COLORS["LR"]),
                                    (-bar_h / 2, "GB", PAPER_COLORS["GB"])]:
            imps = np.array([lookup(exp_id, clf, f)[0] for f in factors])
            stds = np.array([lookup(exp_id, clf, f)[1] for f in factors])
            ax.barh(ypos + offset, imps, height=bar_h,
                    xerr=stds, color=color, edgecolor="white",
                    error_kw=dict(elinewidth=0.7, ecolor="0.3", capsize=2))

        ax.set_yticks(ypos)
        ax.set_yticklabels([PRETTY.get(f, f) for f in factors])
        ax.set_xlabel("Permutation importance")
        ax.set_title(title, loc="left")
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlim(0, 0.36)
        # Light grid only on x.
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    # Single shared legend at the top.
    handles = [mpatches.Patch(facecolor=PAPER_COLORS["LR"], label="Logistic regression"),
               mpatches.Patch(facecolor=PAPER_COLORS["GB"], label="Gradient boosting")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "paper", "plots", "exp1_importance.pdf",
    )
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


main()