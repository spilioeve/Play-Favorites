from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from .config import MODEL_ORDER, FAMILY_ORDER, PLOTS_DIR
from .stats_models import goodman_kruskal_gamma

sns.set_theme(style="whitegrid")

def heatmap_human_vs_llm(
    df: pd.DataFrame,
    outfile_prefix: str = "heatmap_human_vs_llm_scores",
    drop_empty: bool = True,
    annotate: bool = True,
    decimals: int = 2,
):
    """Heatmap of judge (rows) vs model (cols).
    - drop_empty: remove judges/models with no data before plotting
    - annotate: print numeric values in each cell (raw mean scores)
    - decimals: number of decimals for annotations
    """
    # Mean rating per judge→model (LLM judges)
    dfp = (
        df.groupby(["judge", "model"], as_index=False)["rating"].mean()
          .rename(columns={"rating": "scores"})
    )
    # Normalize within judge for the color scale
    dfp["normalized_scores"] = dfp.groupby("judge")["scores"].transform(
        lambda s: (s - s.mean()) / (s.max() - s.min() if s.max() > s.min() else 1)
    )

    # Human row: mean(gt) by model
    human = (
        df.groupby("model", as_index=False)["gt"].mean()
          .rename(columns={"gt": "scores"})
    )
    human["judge"] = "Human"
    denom = (human["scores"].max() - human["scores"].min()) or 1
    human["normalized_scores"] = (human["scores"] - human["scores"].mean()) / denom

    dfp = pd.concat([dfp, human], ignore_index=True)

    # Pivot both: color matrix (normalized) and label matrix (raw)
    mat_val = dfp.pivot_table(index="judge", columns="model", values="normalized_scores")
    mat_lab = dfp.pivot_table(index="judge", columns="model", values="scores")

    # Optionally drop judges/models that are entirely NaN (no data)
    if drop_empty:
        row_mask = mat_val.notna().any(axis=1)
        col_mask = mat_val.notna().any(axis=0)
        mat_val = mat_val.loc[row_mask, col_mask]
        mat_lab = mat_lab.loc[mat_val.index, mat_val.columns]

    # Bail gracefully if nothing left
    if mat_val.shape[0] == 0 or mat_val.shape[1] == 0:
        print("[WARN] heatmap_human_vs_llm: nothing to plot after drop_empty; skipping.")
        return

    # Order rows/cols by preferred order, but keep only those present
    y_order = [x for x in ["Human"] + MODEL_ORDER if x in mat_val.index]
    x_order = [x for x in MODEL_ORDER if x in mat_val.columns]
    mat_val = mat_val.reindex(index=y_order, columns=x_order)
    mat_lab = mat_lab.reindex(index=y_order, columns=x_order)

    # Build annotation strings (raw scores); hide NaNs
    if annotate:
        annot_array = mat_lab.round(decimals).astype(str)
        annot_array = annot_array.where(~mat_val.isna(), "")
    else:
        annot_array = False

    for size, suffix in [((6, 5), ""), ((9, 3), "_wide")]:
        plt.figure(figsize=size)
        ax = sns.heatmap(
            mat_val,
            cmap="RdYlGn",
            vmin=-1, vmax=1,
            cbar=False,
            linewidths=0.5, linecolor="white",
            annot=annot_array if annotate else False,
            fmt="",
            annot_kws={"fontsize": 8},
        )
        ax.set_xlabel("Model"); ax.set_ylabel("Judge")
        plt.xticks(rotation=90); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{outfile_prefix}{suffix}.pdf")
        plt.close()


def correlation_by_dimension_family(
    df: pd.DataFrame,
    method: str,
    outfile: str,
    legend_loc: str = "upper left",      # "upper left" or "lower right"
    legend_anchor: tuple = (0.01, 0.98), # tweak to (0.99, 0.02) for bottom-right
):
    rows = []
    for (dim, judge, fam), g in df.groupby(["dimension", "judge", "judge_family"], dropna=False):
        x = g["rating"].values; y = g["gt"].values
        if method == "pearson":
            val = np.corrcoef(x, y)[0, 1]
        elif method == "spearman":
            from scipy.stats import spearmanr
            val = spearmanr(x, y, nan_policy="omit").correlation
        elif method == "kendall":
            from scipy.stats import kendalltau
            val = kendalltau(x, y, nan_policy="omit").correlation
        elif method == "gamma":
            val = goodman_kruskal_gamma(x, y)
        else:
            raise ValueError(method)
        rows.append({"dimension": dim, "judge": judge, "judge_family": fam, "score": val})
    corrs = pd.DataFrame(rows)

    g = sns.catplot(
        data=corrs, kind="bar", x="judge", y="score", hue="judge_family",
        col="dimension", col_wrap=3, sharex=False, sharey=True, height=3
    )
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=90)
    g.set_axis_labels("Judge", "Correlation")

    # ---- Legend: move outside figure to avoid overlapping facets ----
    # capture handles/labels from any axis BEFORE removing seaborn’s legend
    handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
    if g._legend is not None:
        g._legend.remove()
    # create a single figure-level legend
    if handles:
        g.figure.legend(
            handles, labels,
            loc=legend_loc,           # "upper left" or "lower right"
            bbox_to_anchor=legend_anchor,
            frameon=False,
            title="Judge family",
        )

    # leave some breathing room for the external legend
    g.figure.subplots_adjust(top=0.96, right=0.98, left=0.08, bottom=0.3, wspace=0.15, hspace=1.40)

    g.figure.savefig(PLOTS_DIR / outfile)
    plt.close(g.figure)


def point_range_plot(
    df: pd.DataFrame,
    x_order: list,
    ylab: str,
    outfile: str,
    xlim: tuple[float, float] | None = None,
    debug: bool = False,
):
    # Clean + ensure string labels
    d = df.dropna(subset=["term", "estimate"]).copy()
    d["term"] = d["term"].astype(str)

    # union y-order: (respect MODEL_ORDER first) + any extras we actually have
    seen_terms = list(d["term"].unique())
    y_order = [m for m in x_order if m in set(seen_terms)] + [t for t in seen_terms if t not in set(x_order)]

    # map each term to a numeric y position
    y_index = {t: i for i, t in enumerate(y_order)}
    d["ypos"] = d["term"].map(y_index)

    # fallback err if missing (so plotting never crashes)
    if "err" not in d or d["err"].isna().all():
        d["err"] = 0.0

    # choose colors per Family (or single color if Family is missing)
    families = [f for f in d["Family"].dropna().unique()] if "Family" in d else []
    palette = sns.color_palette(n_colors=max(1, len(families)))

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    if families:
        for c, fam in zip(palette, families):
            sub = d[d["Family"] == fam]
            ax.scatter(sub["estimate"], sub["ypos"], s=40, label=str(fam), color=c)
            # horizontal error segments
            for _, r in sub.iterrows():
                ax.hlines(y=r["ypos"],
                          xmin=r["estimate"] - r["err"],
                          xmax=r["estimate"] + r["err"],
                          lw=1, color=c)
        ax.legend(title="Family", loc="best", frameon=False)
    else:
        # no Family column or all NaN → single color
        ax.scatter(d["estimate"], d["ypos"], s=40)
        for _, r in d.iterrows():
            ax.hlines(y=r["ypos"],
                      xmin=r["estimate"] - r["err"],
                      xmax=r["estimate"] + r["err"],
                      lw=1)

    # y ticks & labels as categorical judge names
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order)

    # x-limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        if not d.empty:
            xmin = float(min(0.0, d["estimate"].min()))
            xmax = float(max(0.0, d["estimate"].max()))
            span = xmax - xmin
            min_width = 0.05  # good for ~0.01–0.04 effects
            if span < min_width:
                center = (xmin + xmax) / 2.0
                half = min_width / 2.0
                xmin, xmax = center - half, center + half
            else:
                pad = span * 0.1
                xmin, xmax = xmin - pad, xmax + pad
            ax.set_xlim(xmin, xmax)

    ax.axvline(0, ls="--", c="k", alpha=0.3)
    ax.set_ylabel("Judge")
    ax.set_xlabel(ylab)

    if debug:
        missing_in_order = sorted(set(seen_terms) - set(x_order))
        print(f"[DEBUG] {outfile}: points={len(d)}, xlim={ax.get_xlim()}, "
              f"terms_not_in_MODEL_ORDER={missing_in_order}")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / outfile)
    plt.close()



def self_bias_by_dimension_plot(
    df: pd.DataFrame,
    model_order: list,
    dimension_order: list,
    outfile: str,
    xlim: tuple[float, float] | None = None,
    debug: bool = False,
    dodge: float = 0.18,
    legend_title: str = "Dimension",
):
    """
    Plot multiple self-bias estimates per judge (one point per dimension) with CIs.
    df requires: ['term' (judge), 'estimate', 'err', 'Factor', 'Family'(opt.)]
    """
    if df.empty:
        print("[WARN] self_bias_by_dimension_plot: empty dataframe; skipping.")
        return

    d = df.copy()
    d["term"] = d["term"].astype(str).str.replace(r"^T\.", "", regex=True)

    # y positions (judges)
    seen_terms = list(d["term"].unique())
    y_order = [m for m in model_order if m in set(seen_terms)] + [t for t in seen_terms if t not in set(model_order)]
    y_index = {t: i for i, t in enumerate(y_order)}
    d["ybase"] = d["term"].map(y_index)

    # which dimensions appear? keep your preferred order
    dims_present = [x for x in dimension_order if x in set(d["Factor"])]
    if not dims_present:
        dims_present = sorted(d["Factor"].dropna().unique())
    k = len(dims_present)
    if k == 0:
        print("[WARN] self_bias_by_dimension_plot: no Factor levels; skipping.")
        return
    offsets = {dim: (i - (k - 1) / 2.0) * dodge for i, dim in enumerate(dims_present)}

    # shape for each dimension
    base_markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<"]
    marker_map = {dim: base_markers[i % len(base_markers)] for i, dim in enumerate(dims_present)}

    # color by family (optional)
    families = sorted(d["Family"].dropna().unique()) if "Family" in d else []
    palette = dict(zip(families, sns.color_palette(n_colors=max(1, len(families)))))

    plt.figure(figsize=(7.2, 4.6))
    ax = plt.gca()

    for dim in dims_present:
        sub = d[d["Factor"] == dim]
        if sub.empty:
            continue
        ys = sub["ybase"].values + offsets[dim]
        xs = sub["estimate"].values
        errs = sub.get("err", pd.Series(np.zeros(len(sub)))).values
        cols = [palette.get(f, None) for f in sub["Family"].values] if families else None

        ax.scatter(xs, ys, s=42, marker=marker_map[dim], label=dim, c=cols)
        ax.errorbar(xs, ys, xerr=errs, fmt="none", ecolor="black", elinewidth=1, capsize=2)

    # y ticks only at base (one per judge)
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order)

    # x-limits (per-dimension effects can be bigger than global)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        xmin = float(min(0.0, (d["estimate"] - d.get("err", 0)).min()))
        xmax = float(max(0.0, (d["estimate"] + d.get("err", 0)).max()))
        span = xmax - xmin
        min_width = 0.10
        if span < min_width:
            c = (xmin + xmax) / 2.0; half = min_width / 2.0
            xmin, xmax = c - half, c + half
        else:
            pad = span * 0.1
            xmin, xmax = xmin - pad, xmax + pad
        ax.set_xlim(xmin, xmax)

    ax.axvline(0, ls="--", c="k", alpha=0.3)
    ax.set_ylabel("Judge")
    ax.set_xlabel("Estimate of self-bias")

    # show a legend for Factor (marker shapes)
    handles = [mpl.lines.Line2D([0], [0], marker=marker_map[dv], color='w',
                                markerfacecolor='black', markersize=6, linestyle='', label=dv)
               for dv in dims_present]
    
    fig = ax.figure
    # leave a bit of room so the legend doesn't clip; adjust if needed
    fig.subplots_adjust(bottom=0.37, top = 0.95, right=0.95, left = 0.2)
    fig.legend(
        handles=handles,
        labels=dims_present,
        title=legend_title,
        frameon=False,
        loc="lower right", 
        ncol=2,         # bottom-right corner
        bbox_to_anchor=(0.5, 0.04) # nudge inside the figure
    )

    # ax.legend(handles=handles, title="Dimension", frameon=False, loc="lower right", ncol=2)

    # if debug:
    #     print(f"[DEBUG] {outfile}: dims={dims_present}, xlim={ax.get_xlim()}, judges={len(y_order)}")

    #plt.tight_layout()
    plt.savefig(PLOTS_DIR / outfile, bbox_inches="tight", pad_inches=0.06)
    plt.savefig(PLOTS_DIR / outfile)
    plt.close()


