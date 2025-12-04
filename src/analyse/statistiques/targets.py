# src/feature_analysis/targets.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.analyse.dataset.cibles import TargetSummaryForLLM

from .config import FEAnalysisConfig


def analyze_targets(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    config: FEAnalysisConfig,
) -> dict[str, Any]:
    """
    Analyse les colonnes cibles d'un DataFrame pour guider le choix du cadre
    (classification / régression) et les étapes de feature engineering autour de y.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données brutes (features + cibles).
    target_cols : Sequence[str]
        Liste des noms de colonnes cibles à analyser.
    config : FEAnalysisConfig
        Objet de configuration contenant les seuils (max_classes_for_classif, etc.).

    Returns
    -------
    result : Dict[str, Any]
        {
          "summary": {nom_cible -> dict d'informations et de recommandations},
          "llm":     {nom_cible -> TargetSummaryForLLM}
        }
    """
    summary: dict[str, Any] = {}
    llm_targets: dict[str, TargetSummaryForLLM] = {}

    n_rows = len(df)

    for t in target_cols:
        if t not in df.columns:
            raise ValueError(f"Cible '{t}' absente des colonnes du DataFrame.")

        s = df[t]
        n_unique = s.nunique(dropna=True)
        missing_rate = float(s.isna().mean())
        dtype = str(s.dtype)

        target_type = "unknown"
        problem_hint = "unknown"
        notes: list[str] = []
        value_counts = None
        stats = None

        # --- 1) Détermination du type de problème (classification / régression) ---
        is_numeric = pd.api.types.is_numeric_dtype(s)

        if is_numeric:
            # Heuristique : classification vs régression
            if n_unique <= 2:
                target_type = "numeric-binary"
                problem_hint = "binary_classification"
            elif n_unique <= config.max_classes_for_classif and (n_unique / max(1, n_rows) < 0.2):
                # Probable multi-classe
                target_type = "numeric-few-classes"
                problem_hint = "multiclass_classification"
            else:
                target_type = "numeric-continuous"
                problem_hint = "regression"
        else:
            # Non numérique => classification
            if n_unique <= 2:
                target_type = "categorical-binary"
                problem_hint = "binary_classification"
            elif n_unique <= config.max_classes_for_classif:
                target_type = "categorical-few-classes"
                problem_hint = "multiclass_classification"
            else:
                target_type = "categorical-many-classes"
                problem_hint = "high_cardinality_classification"

        # --- 2) Analyse détaillée selon le type de problème ---
        if "classification" in problem_hint:
            # Distribution des classes / modalités
            vc = s.value_counts(dropna=False)
            vc_norm = s.value_counts(normalize=True, dropna=False)
            value_counts = pd.DataFrame({"count": vc, "proportion": vc_norm}).head(30)

            max_prop = vc_norm.max() if len(vc_norm) > 0 else 0.0
            if max_prop > 0.90:
                notes.append("Forte imbalance de classe (une classe > 90%).")
            elif max_prop > 0.80:
                notes.append("Imbalance de classe importante (une classe > 80%).")

        elif problem_hint == "regression":
            # Statistiques descriptives + skew
            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            skew = s.dropna().skew() if s.notna().any() else np.nan
            stats = {
                "describe": desc,
                "skew": skew,
            }
            if np.isfinite(skew) and abs(skew) > 1:
                notes.append(
                    f"Distribution très asymétrique (skew={skew:.2f}) : "
                    "penser à des transformations (log, box-cox, etc.)."
                )

        # --- 3) Valeurs manquantes ---
        if missing_rate > 0:
            notes.append(f"Valeurs manquantes : {missing_rate:.1%} (à gérer).")

        # --- 4) Construction du dict "classique" pour cette cible ---
        info: dict[str, Any] = {
            "dtype": dtype,
            "n_unique": int(n_unique),
            "missing_rate": missing_rate,
            "target_type": target_type,
            "problem_hint": problem_hint,
            "value_counts": value_counts,
            "stats": stats,
            "notes": notes,
        }
        summary[t] = info

        # --- 5) Construction de la dataclass TargetSummaryForLLM pour le LLM ---
        # inferred_target_type pour la dataclass
        if is_numeric:
            inferred_target_type = "numeric"
        else:
            inferred_target_type = "categorical"

        # on mappe problem_hint -> problem_type "standardisé" pour la dataclass
        if problem_hint == "regression":
            problem_type = "regression"
        elif problem_hint == "binary_classification":
            problem_type = "binary_classification"
        elif problem_hint in {"multiclass_classification", "high_cardinality_classification"}:
            problem_type = "multiclass_classification"
        else:
            problem_type = "unknown"

        class_counts = None
        class_proportions = None
        most_frequent_classes = None
        imbalance_ratio = None
        is_imbalanced = None
        positive_class = None

        if "classification" in problem_hint:
            vc_classes = s.value_counts(dropna=True)
            total = float(vc_classes.sum()) if vc_classes.sum() > 0 else 1.0

            class_counts = {str(k): int(v) for k, v in vc_classes.items()}
            class_proportions = {str(k): float(v / total) for k, v in vc_classes.items()}
            most_frequent_classes = [
                {"value": str(k), "count": int(v), "freq": float(v / total)}
                for k, v in vc_classes.head(10).items()
            ]

            if len(vc_classes) >= 2:
                maj = float(vc_classes.iloc[0])
                min_ = float(vc_classes.iloc[-1])
                if min_ > 0:
                    imbalance_ratio = maj / min_
                    is_imbalanced = imbalance_ratio > 5.0
                    if is_imbalanced:
                        notes.append(
                            f"Cible déséquilibrée (rapport max/min ≈ {imbalance_ratio:.1f})."
                        )

            if problem_hint == "binary_classification" and len(vc_classes) == 2:
                positive_class = str(vc_classes.index[0])
                # on ne double pas forcément la note ici, à toi de voir :
                # notes.append(f"Problème binaire, classe positive supposée: {positive_class!r}.")

        llm_targets[t] = TargetSummaryForLLM(
            name=t,
            problem_type=problem_type,
            pandas_dtype=dtype,
            inferred_target_type=inferred_target_type,
            n_rows=n_rows,
            n_unique=int(n_unique),
            missing_rate=missing_rate,
            class_counts=class_counts,
            class_proportions=class_proportions,
            most_frequent_classes=most_frequent_classes,
            imbalance_ratio=imbalance_ratio,
            is_imbalanced=is_imbalanced,
            positive_class=positive_class,
            notes=list(notes),  # on copie la liste pour la dataclass
        )

    # 👉 on retourne maintenant à la fois :
    # - le dict "summary" pour ton rapport classique
    # - le dict "llm" avec les dataclasses pour le snapshot LLM
    return {
        "summary": summary,
        "llm": llm_targets,
    }
