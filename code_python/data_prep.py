from __future__ import annotations
import re
from typing import Iterable
import numpy as np
import pandas as pd
from .config import DATA_DIR

CLEAN_PATTERNS = [
    r"20241022-",
    r"-20240229",
    r"-instruct-v1:0",
    r"-v2:0",
    r"-v1:0",
    r"-2407-v1:0",
    r"Qwen/",
    r"-instruct-v0:2",
    r":1",
]

RECODE_MAP = {
    "anthropic.claude-v2": "Claude v2",
    "meta.llama3-1-70b": "Llama 3 70B",
    "meta.llama3-1-8b": "Llama 3 8B",
    "mistral.mistral-large-2407": "Mistral Large",
    "mistral.mistral-7b": "Mistral 7B",
    "anthropic.claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "anthropic.claude-3-sonnet": "Claude 3 Sonnet",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-4o": "GPT-4o",
}


def load_data(dimensions: Iterable[str]) -> pd.DataFrame:
    """Load {dimension}.csv from ../data and add `dimension` column."""
    dfs = []
    for dim in dimensions:
        p = DATA_DIR / f"{dim}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")
        df = pd.read_csv(p)
        df["dimension"] = dim
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _clean_name(x: str) -> str:
    if pd.isna(x):
        return x
    y = str(x)
    for pat in CLEAN_PATTERNS:
        y = re.sub(pat, "", y)
    return y.replace("3-5", "3.5")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the R preprocessing decisions.
    Requires columns: judge, model, gt, rating, dataset, prompt_id, pred_length
    """
    df = df.copy()

    # --- per-dimension mean over the union of {gt, rating} (R: mean(unique(c(gt, rating))))
    mean_by_dim = (
        df.groupby("dimension")[['gt', 'rating']]
          .apply(lambda g: np.nanmean(np.unique(np.r_[g['gt'].values, g['rating'].values])))
          .rename("mean_val")
    )
    df = df.merge(mean_by_dim, left_on="dimension", right_index=True, how="left")

    # Keep originals
    df["gt_orig"] = df["gt"]
    df["rating_orig"] = df["rating"]

    # Binary vs mean_val
    df["bin_rating"] = (df["rating"] > df["mean_val"]).astype(int)
    df["bin_gt"] = (df["gt"] > df["mean_val"]).astype(int)

    # --- normalize within dimension by the max over both columns (NOT per-row)
    max_by_dim = (
        df.groupby("dimension")[['gt', 'rating']]
          .apply(lambda g: np.nanmax(np.r_[g['gt'].values, g['rating'].values]))
          .rename("max_val")
    )
    df = df.merge(max_by_dim, left_on="dimension", right_index=True, how="left")
    df.loc[df["max_val"] == 0, "max_val"] = 1.0
    df["gt"] = df["gt"] / df["max_val"]
    df["rating"] = df["rating"] / df["max_val"]

    # Ternary mbin_gt
    df["mbin_gt"] = np.where(df["gt"] > 0.8, 1, np.where(df["gt"] < 0.2, 0, 0.5))

    # Clean names + recode
    df["judge"] = df["judge"].map(_clean_name).replace(RECODE_MAP)
    df["model"] = df["model"].map(_clean_name).replace(RECODE_MAP)

    # same_judge as string label or "0"
    df["same_judge"] = np.where(df["model"] == df["judge"], df["model"], "0")

    def fam(name: str | float) -> str | None:
        if pd.isna(name):
            return None
        name = str(name)
        if "Claude" in name: return "Claude"
        if name.startswith("GPT"): return "GPT"
        if "Mistral" in name: return "Mistral"
        if "Llama 3" in name: return "Llama 3"
        if "Command" in name: return "Command"
        return None

    df["model_family"] = df["model"].map(fam)
    df["judge_family"] = df["judge"].map(fam)

    df["same_family"] = np.where(
        (df["model_family"] == df["judge_family"]) & (df["same_judge"] == "0"),
        df["judge_family"],
        "0",
    )

    # Drop rows with essential NAs
    df = df.dropna(subset=["judge", "model", "gt", "rating"]).reset_index(drop=True)
    return df


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("prompt_id", dropna=False)
    df["length_diff"] = df["pred_length"] - g["pred_length"].transform("mean")
    std = g["pred_length"].transform("std").replace(0, np.nan)
    df["length_feature"] = np.tanh(df["length_diff"] / std)
    df["length_bin"] = (df["pred_length"] > g["pred_length"].transform("mean")).astype(int)
    df = df.dropna(subset=["length_feature"])  # mirror R filter
    return df