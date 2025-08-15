from __future__ import annotations
import pandas as pd
import re
from pathlib import Path
from .config import DEFAULT_DIMENSIONS, MODEL_ORDER, FAMILY_ORDER, DIMENSION_ORDER, Z_90
from .data_prep import load_data, preprocess, add_length_features
from .stats_models import ols_robust, coef_table, ordinal_or_logit_by_dimension, gam_model
from .viz import heatmap_human_vs_llm, correlation_by_dimension_family, point_range_plot, self_bias_by_dimension_plot

# Helper methods

def _clean_treatment_label(s):
    if s is None: return s
    s = str(s)
    m = re.match(r'^C\(\s*same_judge\s*\)\[T\.(.+?)\]$', s)
    if m: return m.group(1)
    m = re.match(r'^same_judge\[T\.(.+?)\]$', s)
    if m: return m.group(1)
    s = re.sub(r'^T\.', '', s)
    s = re.sub(r'^same_judge:?','', s).strip('[]')
    return s


# 1) Load & preprocess
df = load_data(DEFAULT_DIMENSIONS)
df = preprocess(df)

# families df for joins
families = df[["judge_family", "judge"]].drop_duplicates().rename(columns={"judge_family": "family"})

# 2) Heatmaps
heatmap_human_vs_llm(df)

# 3) Correlations
for method in ["pearson", "gamma", "spearman"]:
    correlation_by_dimension_family(df, method, outfile=f"correlation_{method}.pdf")

# 4) Global linear model
mod = ols_robust("rating ~ judge + gt:judge + same_judge + same_family + dimension - 1", data=df)
res = coef_table(mod)

judge_est = res[res.term.str.startswith("judge") & ~res.term.str.contains(":gt")].copy()
judge_est["term"] = judge_est["term"].str.replace("^judge", "", regex=True)

gt_est = res[res.term.str.contains(":gt")].copy()
gt_est["term"] = gt_est["term"].str.replace(":gt", "", regex=False)

gt_est["term"] = gt_est["term"].str.replace("^judge", "", regex=True)

sb_est = res[res.term.str.contains("same_judge")].copy()
sb_est["term"] = sb_est["term"].str.replace("same_judge", "", regex=False)

sf_est = res[res.term.str.contains("same_family")].copy()
sf_est["term"] = sf_est["term"].str.replace("same_family", "", regex=False)

# 5) Self-bias plot
sb_plot = sb_est.merge(families.rename(columns={"judge": "term"}), on="term", how="left").dropna(subset=["term"]).copy()
sb_plot["Family"] = pd.Categorical(sb_plot["family"], categories=FAMILY_ORDER, ordered=True)
sb_plot["err"] = Z_90 * sb_plot["std.error"]


point_range_plot(sb_plot, x_order=MODEL_ORDER, ylab="Estimate of self-bias", xlim=(-0.05, 0.05), outfile="self_preference_bias.pdf")

# 6) Family-bias plot
sf_plot = sf_est.copy()
sf_plot["Family"] = pd.Categorical(sf_plot["term"], categories=FAMILY_ORDER, ordered=True)
sf_plot["err"] = Z_90 * sf_plot["std.error"]
point_range_plot(sf_plot.rename(columns={"term": "term"}), x_order=FAMILY_ORDER, ylab="Estimate of family-bias", outfile="family_preference_bias.pdf")

# 7) Self-bias by dimension (linear)

# Fit per-dimension models and collect coefficients
all_linear_res = []
for dim, d in df.groupby("dimension"):
    mod_d = ols_robust("rating ~ judge + gt:judge + same_judge + same_family - 1", data=d)
    ct = coef_table(mod_d)
    ct["dimension"] = dim
    all_linear_res.append(ct)
all_linear_res = pd.concat(all_linear_res, ignore_index=True)

# Extract same_judge terms and clean labels
sb_dim = all_linear_res[all_linear_res.term.str.contains("same_judge", na=False)].copy()
sb_dim["term"] = sb_dim["term"].map(_clean_treatment_label)
sb_dim = sb_dim[sb_dim["term"].notna() & (sb_dim["term"] != "0")].copy()

# attach CI half-width
sb_dim["err"] = Z_90 * sb_dim["std.error"]

# attach Family (no inner-join that could drop rows)
families_map = dict(zip(families["judge"], families["family"]))
sb_dim["Family"] = sb_dim["term"].map(families_map)

# rename for plot
sb_dim.rename(columns={"dimension": "Factor"}, inplace=True)

if sb_dim.empty:
    print("[DEBUG] sb_dim is empty. Raw term labels were:",
          sorted(all_linear_res.term[all_linear_res.term.str.contains("same_judge", na=False)].unique()))
else:
    # this produces the R-like multi-point plot (one marker per dimension per judge)
    self_bias_by_dimension_plot(
        sb_dim,
        model_order=MODEL_ORDER,
        dimension_order=DIMENSION_ORDER,
        outfile="self_preference_bias_linear_by_dimension.pdf",
        xlim=None,       # or set e.g. (-0.15, 0.05) to match your R figure exactly
        debug=True,
        dodge=0.18,
    )


# -------------------------------
# 8) Regression by Dataset (OLS) — one PDF per dataset
# -------------------------------
fam_map = dict(zip(families["judge"], families["family"]))

rows = []
for ds, dset in df.groupby("dataset"):
    # Control for dimension in-model => estimates reflect an average across dimensions
    mod = ols_robust("rating ~ judge + gt:judge + same_judge + same_family + dimension - 1", data=dset)
    ct = coef_table(mod)

    sb = ct[ct.term.str.contains("same_judge", na=False)].copy()
    sb["term"] = sb["term"].map(_clean_treatment_label)
    sb = sb[sb["term"].notna() & (sb["term"] != "0")]
    sb["err"] = Z_90 * sb["std.error"]
    sb["Dataset"] = ds
    sb["Family"] = sb["term"].map(fam_map)

    rows.append(sb[["term", "estimate", "err", "Dataset", "Family"]])

sb_by_ds = pd.concat(rows, ignore_index=True)

# Order datasets the way you want them to appear in the legend (optional)
DATASET_ORDER = ["chatbotarena", "cnn", "helm-instruct", "mtbench", "stanford", "xsum"]
present = [d for d in DATASET_ORDER if d in set(sb_by_ds["Dataset"])]
if not present:
    present = sorted(sb_by_ds["Dataset"].unique())

# Reuse the multi-point plotter: treat Dataset like "Factor"
sb_for_plot = sb_by_ds.rename(columns={"Dataset": "Factor"})
self_bias_by_dimension_plot(
    sb_for_plot,
    model_order=MODEL_ORDER,
    dimension_order=present,
    outfile="self_preference_bias_by_dataset.pdf",
    xlim=None,               # or set (-0.10, 0.05) to mimic your R window exactly
    debug=True,
    dodge=0.18,
    legend_title="Dataset",  # <— legend reads “Dataset”
)


# 9) Ordinal/logit by dimension
ord_res = ordinal_or_logit_by_dimension(df.assign(rating=df["rating_orig"]))
# (Plotting omitted for brevity.)

# 10) GAM (optional)
sb_gam, sf_gam = gam_model(df)

# 11) Length control
df_len = add_length_features(df)
mod_len = ols_robust("rating ~ judge + gt:judge + same_judge + same_family + dimension + judge:length_feature - 1", data=df_len)
res_len = coef_table(mod_len)

# Compare length-controlled vs baseline (left to user to extend plotting)
print("Analysis complete. Figures saved to ./plots")
