# src/feature_analysis/leakage.py
from collections.abc import Sequence
from typing import Any

import pandas as pd

# 👇 nouvelle dataclass importée
from src.analyse.dataset.all import LeakageSignalForLLM

from .config import FEAnalysisConfig


def detect_leakage(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    config: FEAnalysisConfig,
) -> dict[str, Any]:
    """
    Détection simple de fuites potentielles par corrélation forte entre
    features numériques et cibles numériques.

    Returns
    -------
    result : Dict[str, Any]
        {
          "summary": [  # pour le rapport humain / legacy
             {"feature": ..., "target": ..., "correlation": ..., "note": ...},
             ...
          ],
          "llm": [      # version dataclass pour le snapshot LLM
             LeakageSignalForLLM(...),
             ...
          ]
        }
    """
    summary: list[dict[str, Any]] = []
    llm_signals: list[LeakageSignalForLLM] = []

    # --- 1) Sélection des colonnes numériques ---
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    numeric_targets = [t for t in target_cols if pd.api.types.is_numeric_dtype(df[t])]

    if not numeric_features or not numeric_targets:
        # Pas de numérique -> pas de corrélation à calculer
        return {
            "summary": summary,
            "llm": llm_signals,
        }

    # --- 2) Matrice de corrélation ---
    corr_matrix = df[numeric_features + numeric_targets].corr(numeric_only=True)

    # --- 3) Parcours des couples (feature, target) ---
    for t in numeric_targets:
        for c in numeric_features:
            if c == t:
                continue
            corr = corr_matrix.loc[c, t]
            if pd.notna(corr) and abs(corr) >= config.strong_corr_threshold:
                note = "Corrélation très forte, possible fuite ou duplication de la cible."

                # Vue dict (comme avant)
                summary.append(
                    {
                        "feature": c,
                        "target": t,
                        "correlation": float(corr),
                        "note": note,
                    }
                )

                # Vue dataclass pour le LLM
                llm_signals.append(
                    LeakageSignalForLLM(
                        feature=c,
                        target=t,
                        correlation=float(corr),
                        note=note,
                        severity="strong",
                    )
                )

    return {
        "summary": summary,
        "llm": llm_signals,
    }
