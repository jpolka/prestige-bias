"""
analyze_v2.py
=============
Analyzes the benchmark results from run_batch.py (results_v2_<model>.csv).

For each model it reports:
  1. Data quality: failures and refusals, broken down by tier. If failures
     cluster in one tier, the averages for that tier are biased.
  2. Mean scores by tier.
  3. Tier contrasts, computed WITHIN each paper and then averaged:
       highly ranked R1 - RCU   <- the headline number
       highly ranked R1 - other R1, other R1 - RCU
       each tier - none  (none = no affiliation shown, the true baseline)
     Each with a 95% bootstrap confidence interval.
  4. One effect per institution (its score minus the same paper's "none"
     score), so you can see whether one name drives a whole tier.
Then it compares the headline number across models, saves a summary CSV
and draws a chart.

How to run
----------
    python analyze_v2.py                                # every results_v2_claude-*.csv
    python analyze_v2.py results_v2_claude-haiku-5-5.csv

Missing data rule (failed or refused replies): a paper's tier score is the
mean of whichever institutions in that tier were scored. A paper is left out
of a contrast only if one of the two tiers has no score at all for it.
"""

import glob
import sys

import numpy as np
import pandas as pd

# Model results are named results_v2_claude-<model>.csv. (A looser pattern
# would also pick up results_v2_summary.csv, which this script writes.)
FILES = sys.argv[1:] or sorted(glob.glob("results_v2_claude-*.csv"))
SCORE_KEYS = ["rigor", "novelty", "impact", "overall"]
TIER_ORDER = ["high", "mid", "low", "fictional", "none"]

# The data files use short tier codes (high/mid/low), which are baked into
# the submitted batches. These readable labels are used only for printing and
# the chart, so they can be changed freely without re-running anything.
# Carnegie 2025 research designations: all "high" and "mid" institutions are
# R1; all "low" ones are RCU (Research Colleges and Universities, below R2).
TIER_LABELS = {
    "high": "Highly ranked R1",
    "mid": "Other R1",
    "low": "RCU",
    "fictional": "Fictional",
    "none": "No affiliation",
}

# (tier A, tier B): each contrast is "A minus B". The first is the headline.
CONTRAST_PAIRS = [
    ("high", "low"),
    ("high", "mid"),
    ("mid", "low"),
    ("high", "none"),
    ("mid", "none"),
    ("low", "none"),
    ("fictional", "none"),
]
# Which scores the chart shows. The README chart uses only "overall";
# add "rigor", "novelty", "impact" to see all four per institution.
CHART_SCORES = ["overall"]

CONTRASTS = [("{} - {}".format(TIER_LABELS[a], TIER_LABELS[b]), a, b) for a, b in CONTRAST_PAIRS]
HEADLINE = CONTRASTS[0][0]   # "Highly ranked R1 - RCU"

# Bootstrap: resample PAPERS (with replacement) many times and recompute the
# mean difference each time. The middle 95% of those means is the 95%
# confidence interval: the range of plausible values given which papers we
# happened to sample. A fresh fixed seed for every interval, so each number
# is reproducible and doesn't change when other result files are added.
N_BOOT = 10000
SEED = 42


def bootstrap_ci(values):
    """95% confidence interval for the mean of a per-paper array."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    means = arr[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def load(path):
    df = pd.read_csv(path)
    # The "none" condition has a blank institution; give it a readable label.
    df["label"] = df["institution"].fillna("(no affiliation)")
    return df


def tier_table(ok, key):
    """One row per paper, one column per tier: mean score in that tier."""
    return ok.groupby(["doi", "tier"])[key].mean().unstack().reindex(columns=TIER_ORDER)


def contrast(ok, key, tier_a, tier_b):
    """Per-paper difference (tier_a - tier_b), with papers lacking either dropped."""
    per_paper = tier_table(ok, key)
    return (per_paper[tier_a] - per_paper[tier_b]).dropna()


def institution_effects(ok, key="overall"):
    """For each institution: mean over papers of (its score - that paper's 'none' score)."""
    wide = ok.pivot_table(index="doi", columns="label", values=key)
    baseline = wide.pop("(no affiliation)")
    tiers = ok.drop_duplicates("label").set_index("label")["tier"]
    rows = []
    for label in wide.columns:
        d = (wide[label] - baseline).dropna()
        lo, hi = bootstrap_ci(d)
        rows.append({"institution": label, "tier": tiers[label], "score": key,
                     "effect": d.mean(), "ci_low": lo, "ci_high": hi, "n": len(d)})
    out = pd.DataFrame(rows)
    out["tier"] = pd.Categorical(out["tier"], TIER_ORDER)
    return out.sort_values(["tier", "effect"], ascending=[True, False])


def analyze_model(path):
    df = load(path)
    model = df["model"].iloc[0]
    print("\n" + "=" * 70)
    print("MODEL: {}   ({})".format(model, path))
    print("=" * 70)

    # 1. DATA QUALITY -------------------------------------------------------
    failed = df[df["parse_ok"] != True]
    print("\nRows: {}   Failed: {}".format(len(df), len(failed)))
    if len(failed):
        print("Failures by tier and type:")
        print(failed.assign(tier=failed["tier"].map(TIER_LABELS))
              .groupby(["tier", "error"]).size().to_string())

    ok = df[df["parse_ok"] == True].copy()
    for k in SCORE_KEYS:
        ok[k] = ok[k].astype(int)

    # 2. MEANS BY TIER ------------------------------------------------------
    print("\n--- Mean scores by tier ---")
    means = ok.groupby("tier")[SCORE_KEYS].mean().reindex(TIER_ORDER)
    means.index = [TIER_LABELS[t] for t in means.index]
    print(means.round(2).to_string())

    # 3. CONTRASTS ----------------------------------------------------------
    summary = []
    for key in SCORE_KEYS:
        for name, a, b in CONTRASTS:
            d = contrast(ok, key, a, b)
            lo, hi = bootstrap_ci(d)
            summary.append({
                "model": model, "score": key, "contrast": name,
                "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi,
                "n_papers": len(d),
                "papers_up": int((d > 0).sum()), "papers_down": int((d < 0).sum()),
                "papers_tied": int((d == 0).sum()),
            })
    summary = pd.DataFrame(summary)

    print("\n--- Tier contrasts on 'overall' (within-paper, A minus B) ---")
    print("A positive number means tier A scored higher.")
    print("The CI 'excludes 0' when the whole interval is on one side of zero.\n")
    o = summary[summary["score"] == "overall"]
    for r in o.itertuples():
        excl = "  <- CI excludes 0" if (r.ci_low > 0 or r.ci_high < 0) else ""
        print("  {:<35} {:+.3f}  [{:+.3f}, {:+.3f}]  n={}  up/down/tied {}/{}/{}{}".format(
            r.contrast, r.mean_diff, r.ci_low, r.ci_high, r.n_papers,
            r.papers_up, r.papers_down, r.papers_tied, excl))

    print("\n--- Headline ({}) for every criterion ---".format(HEADLINE))
    h = summary[summary["contrast"] == HEADLINE]
    for r in h.itertuples():
        print("  {:<8} {:+.3f}  [{:+.3f}, {:+.3f}]".format(r.score, r.mean_diff, r.ci_low, r.ci_high))

    # 4. PER-INSTITUTION EFFECTS -------------------------------------------
    print("\n--- Per-institution effect on 'overall' vs. no affiliation ---")
    # Computed for all four scores (the chart shows them all); printed for 'overall'.
    inst = pd.concat([institution_effects(ok, k) for k in SCORE_KEYS])
    for r in inst[inst["score"] == "overall"].itertuples():
        print("  {:<17} {:<42} {:+.3f}  [{:+.3f}, {:+.3f}]".format(
            TIER_LABELS[r.tier], r.institution, r.effect, r.ci_low, r.ci_high))
    inst["model"] = model

    return summary, inst


def draw_chart(inst_all, path="results_v2_chart.png"):
    """One panel per model; each row an institution; one dot per score in CHART_SCORES.

    Each dot is the institution's mean effect on that score vs. the same
    paper with no affiliation, with its 95% CI. With several scores, they sit
    just above/below each other within the institution's row. Tiers are shown
    by light background bands, labelled on the right.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Score colors: indigo / teal / crimson / amber. Checked with a Python
    # port of the dataviz skill's palette validator: every pair stays
    # distinguishable under simulated protan/deutan/tritan color blindness
    # (OKLab Delta E >= 8), every color has >= 3:1 contrast on all four
    # background tints, and each score also gets its own marker shape.
    score_style = {
        "overall": dict(color="#3937B8", marker="o", size=7),
        "rigor":   dict(color="#028A7D", marker="s", size=5),
        "novelty": dict(color="#C17808", marker="^", size=5.5),
        "impact":  dict(color="#C43845", marker="D", size=4.5),
    }
    # Vertical position of each plotted score inside a row (row height = 1).
    step = 0.18
    offset = {k: (i - (len(CHART_SCORES) - 1) / 2) * step for i, k in enumerate(CHART_SCORES)}
    # Very light tier backgrounds.
    band_color = {"high": "#EEF2FA", "mid": "#EDF6F0", "low": "#FCF2E8", "fictional": "#F2F2F0"}

    # Panels from smallest to largest model; any model not listed goes last.
    size_order = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-4-7", "claude-opus-5-5"]
    models = sorted(inst_all["model"].unique(),
                    key=lambda m: size_order.index(m) if m in size_order else len(size_order))
    # Fixed institution order (by tier, then name) so rows line up across
    # panels. De-duplicate first: each institution appears once per model and score.
    names = inst_all.drop_duplicates("institution").copy()
    names["tier"] = pd.Categorical(names["tier"].astype(str), TIER_ORDER)
    names = names.sort_values(["tier", "institution"])
    order = names["institution"].tolist()
    tier_of = dict(zip(order, names["tier"].astype(str)))
    y_of = {name: i for i, name in enumerate(order)}

    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models) + 3.2,
                                      (0.26 + 0.09 * len(CHART_SCORES)) * len(order) + 1.8),
                             sharey=True, sharex=True, squeeze=False)  # same x scale: compare models fairly
    axes = axes[0]
    for ax, model in zip(axes, models):
        # Background bands: one span per tier.
        for tier, color in band_color.items():
            rows = [y_of[n] for n in order if tier_of[n] == tier]
            if rows:
                ax.axhspan(min(rows) - 0.5, max(rows) + 0.5, color=color, zorder=0, linewidth=0)
        ax.axvline(0, color="#6b6b66", linewidth=1.1, zorder=2.6)  # above the white gridlines

        sub = inst_all[inst_all["model"] == model]
        for key in CHART_SCORES:
            s = score_style[key]
            t = sub[sub["score"] == key]
            y = t["institution"].map(y_of) + offset[key]
            ax.errorbar(t["effect"], y,
                        xerr=[t["effect"] - t["ci_low"], t["ci_high"] - t["effect"]],
                        fmt=s["marker"], color=s["color"], ecolor=s["color"],
                        markersize=s["size"], markeredgecolor="white", markeredgewidth=0.6,
                        elinewidth=1.4 if key == "overall" else 1.0, alpha=1.0 if key == "overall" else 0.9,
                        capsize=0, label=key.capitalize(), zorder=4 if key == "overall" else 3)

        # Papers in this panel (fewer than 100 if some were refused for every condition).
        n_papers = int(sub[sub["score"] == "overall"]["n"].max())
        ax.set_title("{}\n{} papers".format(model, n_papers), fontsize=11, fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#ffffff", linewidth=0.9, zorder=0.5)  # white gridlines read over bands
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(False)

    # Tier names on the right edge of the last panel, so the band colors
    # are never the only way to tell tiers apart.
    last = axes[-1]
    for tier, label in TIER_LABELS.items():
        rows = [y_of[n] for n in order if tier_of[n] == tier]
        if rows:
            last.text(1.02, (min(rows) + max(rows)) / 2, label, transform=last.get_yaxis_transform(),
                      va="center", ha="left", fontsize=9, color="#444444", fontweight="bold")

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=9)
    axes[0].set_ylim(len(order) - 0.5, -0.5)  # first institution at the top
    if len(CHART_SCORES) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 1.0))
        what, title_y = "each score", 1.04
    else:
        what, title_y = "'{}' score".format(CHART_SCORES[0]), 1.0
    fig.suptitle("Change in {} vs. no affiliation (mean over papers, 95% CI)".format(what),
                 y=title_y, fontsize=12)
    fig.supxlabel("Score difference (points on the 1–10 scale)", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("\nChart saved to " + path)


def main():
    if not FILES:
        sys.exit("No results_v2_*.csv files found.")
    summaries, insts = [], []
    for path in FILES:
        s, i = analyze_model(path)
        summaries.append(s)
        insts.append(i)
    summary = pd.concat(summaries)
    summary.to_csv("results_v2_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("ACROSS MODELS: headline gap ({}) on 'overall'".format(HEADLINE))
    print("=" * 70)
    h = summary[(summary["contrast"] == HEADLINE) & (summary["score"] == "overall")]
    for r in h.itertuples():
        print("  {:<20} {:+.3f}  [{:+.3f}, {:+.3f}]  (n={} papers)".format(
            r.model, r.mean_diff, r.ci_low, r.ci_high, r.n_papers))
    print("\nAll contrasts saved to results_v2_summary.csv")

    draw_chart(pd.concat(insts))


if __name__ == "__main__":
    main()
