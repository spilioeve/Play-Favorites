from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from .config import Z_90

try:
    from pygam import LinearGAM, s, l
    _HAS_PYGAM = True
except Exception:
    _HAS_PYGAM = False



def ols_robust(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(cov_type="HC1")


def coef_table(model) -> pd.DataFrame:
    return (pd.DataFrame({
        "term": model.params.index,
        "estimate": model.params.values,
        "std.error": model.bse.values,
        "statistic": model.tvalues.values,
        "p.value": model.pvalues.values,
    }))


def ordinal_or_logit_by_dimension(dfo: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dim, df_dim in dfo.groupby("dimension"):
        y = df_dim["rating_orig"] if "rating_orig" in df_dim else df_dim["rating"]
        uniq = np.sort(pd.unique(y))
        if len(uniq) > 2:
            # Ordered logit using one-hot for linear terms + interaction gt:judge via multiplication
            X_lin = pd.get_dummies(df_dim[["judge", "same_judge", "same_family"]], drop_first=False)
            X_int = pd.get_dummies(df_dim[["judge"]], prefix="gt:judge").mul(df_dim["gt"].values, axis=0)
            exog = pd.concat([X_lin, X_int], axis=1)
            try:
                res = OrderedModel(y.astype("category"), exog, distr="logit").fit(method="bfgs", disp=False)
            except Exception:
                continue
            params = res.params[~res.params.index.str.contains("threshold")]
            bse = res.bse[params.index]
            tmp = pd.DataFrame({"term": params.index, "estimate": params.values, "std.error": bse.values})
        else:
            try:
                res = smf.logit("C(rating, Treatment(0)) ~ judge + gt:judge + same_judge + same_family", data=df_dim).fit(disp=False)
            except Exception:
                continue
            tmp = coef_table(res)
        tmp["dimension"] = dim
        out.append(tmp)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["term","estimate","std.error","dimension"])


def gam_model(df: pd.DataFrame):
    """
    Fit a GAM with:
      - linear terms for judge/same_judge/same_family/dimension dummies
      - a spline (smooth) on gt
    Return two DataFrames with linear-term coefficients for same_judge and same_family.
    If pygam isn't installed or fitting fails, return (None, None).
    """
    if not _HAS_PYGAM:
        warnings.warn("pygam not installed; skipping GAM.")
        return None, None

    # One-hot encode linear factors
    X_lin = pd.get_dummies(df[["judge", "same_judge", "same_family", "dimension"]], drop_first=False).astype(float)
    lin_cols = list(X_lin.columns)

    # Build full design: [linear dummies..., gt]
    X = np.column_stack([X_lin.values, df[["gt"]].values])
    y = df["rating"].values
    n_lin = X_lin.shape[1]       # number of linear features
    gt_idx = n_lin               # index of gt in the combined matrix

    # Build the GAM term structure: l(0) + l(1) + ... + l(n_lin-1) + s(n_lin)
    term = l(0)
    for i in range(1, n_lin):
        term = term + l(i)
    term = term + s(gt_idx)

    # Fit
    try:
        gam = LinearGAM(term).fit(X, y)
    except Exception as e:
        warnings.warn(f"GAM fit failed: {e}")
        return None, None

    # Map coefficients per term. In pygam, coef_ packs:
    # [intercept] + [1 coef per linear term, in order] + [many spline coefs]
    coefs = gam.coef_
    offset = 1 if gam.fit_intercept else 0

    # Extract linear coefs aligned with lin_cols
    if offset + n_lin > len(coefs):
        # Safety guard: unexpected layout → skip
        warnings.warn("Unexpected GAM coefficient layout; skipping GAM outputs.")
        return None, None

    linear_coefs = coefs[offset : offset + n_lin]
    lin_df = pd.DataFrame({"term": lin_cols, "estimate": linear_coefs})

    # Pull self-bias / family-bias linear terms
    sb = lin_df[lin_df["term"].str.contains("same_judge", na=False)].copy()
    sf = lin_df[lin_df["term"].str.contains("same_family", na=False)].copy()

    # Keep just the *labels* (not one-hot column names) for readability
    # Columns look like 'same_judge_<ModelName>' or 'same_family_<Family>'
    # We strip the prefix so downstream code can treat them like before.
    sb["term"] = sb["term"].str.replace(r"^same_judge_", "", regex=True)
    sf["term"] = sf["term"].str.replace(r"^same_family_", "", regex=True)

    # std.errors for pygam linear coefs aren’t directly exposed like statsmodels;
    # set to NaN so downstream plotting skips intervals (or you can compute via bootstrapping).
    sb["std.error"] = np.nan
    sf["std.error"] = np.nan

    return sb.reset_index(drop=True), sf.reset_index(drop=True)



def goodman_kruskal_gamma(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y); n = len(x)
    C = D = 0
    for i in range(n-1):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        s = np.sign(dx * dy)
        C += np.sum(s > 0)
        D += np.sum(s < 0)
    return np.nan if (C + D) == 0 else (C - D) / (C + D)
