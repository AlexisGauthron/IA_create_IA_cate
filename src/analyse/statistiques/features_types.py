from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd


# 👇 dataclasses pour le LLM
from src.analyse.dataset.features import (
    NumericStats,
    CategoricalStats,
    TextStats,
)


# ----------------------------------------------------------------------
# Helpers pour remplir les stats LLM
# ----------------------------------------------------------------------
def _build_numeric_stats(s: pd.Series) -> NumericStats:
    """Construit NumericStats à partir d'une série numérique."""
    s_num = pd.to_numeric(s, errors="coerce")

    if s_num.dropna().empty:
        return NumericStats()

    desc = s_num.describe(percentiles=[0.25, 0.5, 0.75])
    skew = float(s_num.skew(skipna=True) or 0.0)
    kurt = float(s_num.kurtosis(skipna=True) or 0.0)

    return NumericStats(
        mean=float(desc.get("mean", np.nan)),
        std=float(desc.get("std", np.nan)),
        min=float(desc.get("min", np.nan)),
        p25=float(desc.get("25%", np.nan)),
        p50=float(desc.get("50%", np.nan)),
        p75=float(desc.get("75%", np.nan)),
        max=float(desc.get("max", np.nan)),
        skewness=skew,
        kurtosis=kurt,
    )


def _build_categorical_stats(
    s: pd.Series,
    *,
    n_rows: int,
    n_top: int = 10,
    rare_level_threshold: float = 0.01,
) -> CategoricalStats:
    """Construit CategoricalStats à partir d'une série catégorielle / object."""
    s_obj = s.astype("object")
    vc = s_obj.value_counts(dropna=True)
    n_unique = int(vc.shape[0])
    unique_ratio = float(n_unique / n_rows) if n_rows > 0 else 0.0

    top_values: List[Dict[str, Any]] = []
    for val, count in vc.head(n_top).items():
        freq = float(count / n_rows) if n_rows > 0 else 0.0
        top_values.append(
            {
                "value": str(val),
                "count": int(count),
                "freq": freq,
            }
        )

    if n_rows > 0:
        rare_mask = (vc / n_rows) < rare_level_threshold
        n_rare = int(rare_mask.sum())
    else:
        n_rare = 0

    return CategoricalStats(
        n_unique=n_unique,
        unique_ratio=unique_ratio,
        top_values=top_values,
        n_rare_levels=n_rare,
        rare_level_threshold=rare_level_threshold,
    )


def _build_text_stats(
    s: pd.Series,
    *,
    n_examples: int = 5,
) -> TextStats:
    """Construit TextStats à partir d'une série texte."""
    s_str = s.dropna().astype(str)
    if s_str.empty:
        return TextStats()

    lengths = s_str.str.len()
    avg_char = float(lengths.mean())
    min_char = int(lengths.min())
    max_char = int(lengths.max())

    tok_counts = s_str.str.split().map(len)
    avg_tok = float(tok_counts.mean())


    return TextStats(
        avg_char_length=avg_char,
        avg_token_length=avg_tok,
        min_char_length=min_char,
        max_char_length=max_char,
    )
