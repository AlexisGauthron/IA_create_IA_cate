from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

# Dataclasses pour le LLM
from src.analyse.dataset.features import (
    CategoricalStats,
    FeatureSummaryForLLM,
    NumericStats,
    TextStats,
)
from src.analyse.statistiques.features_types import (
    _build_categorical_stats,
    _build_numeric_stats,
    _build_text_stats,
)

from .config import FEAnalysisConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Fonctions utilitaires (privées)
# ----------------------------------------------------------------------


def _determine_role(
    series: pd.Series,
    n_unique: int,
    unique_ratio: float,
    config: FEAnalysisConfig,
) -> tuple[str, bool]:
    """
    Détermine le rôle logique d'une colonne.

    Returns:
        Tuple[role, is_bool_like]
        - role: "numeric", "categorical", "text", "datetime", "boolean", "unknown"
        - is_bool_like: True si la colonne ressemble à un booléen
    """
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    is_bool_type = pd.api.types.is_bool_dtype(series)

    # Détection bool-like par les valeurs (0/1, True/False)
    non_na_values = series.dropna().unique().tolist()
    is_bool_values = False
    if len(non_na_values) > 0:
        bool_like_set = {0, 1}
        try:
            is_bool_values = set(non_na_values).issubset(bool_like_set)
        except TypeError:
            is_bool_values = False

    is_bool_like = is_bool_type or is_bool_values

    # Détermination du rôle
    if pd.api.types.is_numeric_dtype(series) and not is_bool_like:
        return "numeric", is_bool_like
    elif is_datetime:
        return "datetime", is_bool_like
    elif is_bool_like:
        return "boolean", is_bool_like
    elif pd.api.types.is_categorical_dtype(series.dtype) or series.dtype == "object":
        # Différencier texte libre vs catégorielle
        if (n_unique > config.max_unique_cat_low) and (
            unique_ratio > config.text_unique_ratio_threshold
        ):
            return "text", is_bool_like
        else:
            return "categorical", is_bool_like

    return "unknown", is_bool_like


def _detect_flags(
    col_name: str,
    role: str,
    n_unique: int,
    unique_ratio: float,
    config: FEAnalysisConfig,
) -> tuple[list[str], bool, bool, bool]:
    """
    Détecte les flags utiles pour le FE.

    Returns:
        Tuple[flags, is_constant, is_id_like, high_cardinality]
    """
    is_constant = n_unique <= 1

    is_id_like_name = any(key in col_name.lower() for key in ["id", "uuid", "guid"])
    is_id_like_ratio = unique_ratio > config.id_unique_ratio_threshold
    is_id_like = bool(is_id_like_name or is_id_like_ratio)

    high_cardinality = False
    if role == "categorical" and n_unique > config.high_cardinality_threshold:
        high_cardinality = True

    # Construction des flags
    flags: list[str] = []
    if is_constant:
        flags.append("CONSTANT")
    if is_id_like:
        flags.append("ID_LIKE")
    if high_cardinality:
        flags.append("HIGH_CARDINALITY")

    return flags, is_constant, is_id_like, high_cardinality


def _generate_recommendations(
    series: pd.Series,
    role: str,
    missing_rate: float,
    is_constant: bool,
    is_id_like: bool,
    high_cardinality: bool,
    n_unique: int,
    config: FEAnalysisConfig,
) -> tuple[
    list[str],
    list[str],
    list[str],
    Any,
    NumericStats | None,
    CategoricalStats | None,
    TextStats | None,
]:
    """
    Génère les notes, recommandations et hints FE selon le rôle.

    Returns:
        Tuple[notes, recommendations, fe_hints, extra_info, numeric_stats, categorical_stats, text_stats]
    """
    notes: list[str] = []
    recommendations: list[str] = []
    fe_hints: list[str] = []
    extra_info: Any = None
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    text_stats: TextStats | None = None

    n_rows = len(series)

    # --- Flags généraux ---
    if is_constant:
        notes.append("Variable constante : probablement à supprimer.")
        recommendations.append("Supprimer : aucune information prédictive.")
        fe_hints.append("drop_constant_feature")

    if is_id_like:
        notes.append("Variable probablement identifiant (ID-like).")
        recommendations.append(
            "Ne pas utiliser telle quelle comme feature : risque de fuite ou de sur-apprentissage."
        )
        fe_hints.append("do_not_use_as_raw_feature")

    # --- Valeurs manquantes ---
    if missing_rate > 0:
        notes.append(f"Valeurs manquantes : {missing_rate:.1%}.")
        if missing_rate > config.high_missing_threshold:
            notes.append(
                f"Taux de NaN élevé (> {config.high_missing_threshold:.0%}) : "
                "envisager imputation robuste ou exclusion."
            )
            fe_hints.append("high_missing_rate")

    # --- Recommandations spécifiques par rôle ---
    if role == "numeric":
        notes, recommendations, fe_hints, extra_info, numeric_stats = _recommendations_numeric(
            series, notes, recommendations, fe_hints, missing_rate
        )

    elif role == "categorical":
        notes, recommendations, fe_hints, categorical_stats = _recommendations_categorical(
            series,
            notes,
            recommendations,
            fe_hints,
            missing_rate,
            high_cardinality,
            n_unique,
            n_rows,
            config,
        )

    elif role == "text":
        notes, recommendations, fe_hints, text_stats = _recommendations_text(
            series, notes, recommendations, fe_hints, missing_rate, config
        )

    elif role == "datetime":
        notes, recommendations, fe_hints = _recommendations_datetime(
            notes, recommendations, fe_hints, missing_rate
        )

    elif role == "boolean":
        notes, recommendations, fe_hints = _recommendations_boolean(
            notes, recommendations, fe_hints, missing_rate
        )

    else:
        recommendations.append(
            "Type non clairement identifié : vérifier la sémantique de cette colonne. "
            "Décider si elle doit être traitée comme numérique, catégorielle, texte, etc."
        )
        fe_hints.append("check_semantics")

    return (
        notes,
        recommendations,
        fe_hints,
        extra_info,
        numeric_stats,
        categorical_stats,
        text_stats,
    )


def _recommendations_numeric(
    series: pd.Series,
    notes: list[str],
    recommendations: list[str],
    fe_hints: list[str],
    missing_rate: float,
) -> tuple[list[str], list[str], list[str], dict[str, Any], NumericStats]:
    """Génère recommandations pour une feature numérique."""
    desc = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    skew = series.dropna().skew() if series.notna().any() else np.nan
    extra_info = {"describe": desc, "skew": skew}

    if np.isfinite(skew) and abs(skew) > 1:
        notes.append(f"Distribution très asymétrique (skew={skew:.2f}).")
        recommendations.append(
            "Envisager une transformation (log, box-cox, binning quantile) "
            "pour stabiliser la distribution."
        )
        fe_hints.append("consider_non_linear_transform")

    if missing_rate > 0:
        recommendations.append("Imputation numérique (médiane ou moyenne).")
        fe_hints.append("numeric_imputation")

    recommendations.append(
        "Standardisation ou normalisation recommandée pour modèles linéaires / SVM / KNN."
    )
    recommendations.append("Winsorisation ou clipping possible pour limiter l'impact des outliers.")
    fe_hints.append("candidate_for_scaling")

    numeric_stats = _build_numeric_stats(series)

    return notes, recommendations, fe_hints, extra_info, numeric_stats


def _recommendations_categorical(
    series: pd.Series,
    notes: list[str],
    recommendations: list[str],
    fe_hints: list[str],
    missing_rate: float,
    high_cardinality: bool,
    n_unique: int,
    n_rows: int,
    config: FEAnalysisConfig,
) -> tuple[list[str], list[str], list[str], CategoricalStats]:
    """Génère recommandations pour une feature catégorielle."""
    if high_cardinality:
        notes.append(f"Haute cardinalité : {n_unique} modalités.")
        recommendations.append(
            "Éviter le one-hot naïf. Préférer target encoding, leave-one-out, hashing, "
            "embeddings ou regroupement des modalités rares."
        )
        fe_hints.append("use_target_encoding_or_hashing")
    else:
        recommendations.append("One-hot encoding ou encodage ordinal selon le modèle choisi.")
        fe_hints.append("candidate_for_one_hot")

    if missing_rate > 0:
        recommendations.append(
            "Imputation par catégorie spéciale '__MISSING__' ou valeur NA explicite."
        )
        fe_hints.append("categorical_imputation")

    categorical_stats = _build_categorical_stats(
        series,
        n_rows=n_rows,
        n_top=10,
        rare_level_threshold=getattr(config, "rare_level_threshold", 0.01),
    )

    return notes, recommendations, fe_hints, categorical_stats


def _recommendations_text(
    series: pd.Series,
    notes: list[str],
    recommendations: list[str],
    fe_hints: list[str],
    missing_rate: float,
    config: FEAnalysisConfig,
) -> tuple[list[str], list[str], list[str], TextStats]:
    """Génère recommandations pour une feature texte."""
    notes.append("Colonne texte / semi-structurée.")
    recommendations.append(
        "Feature engineering texte : TF-IDF, n-grams, embeddings (SentenceTransformers), "
        "mots-clés, longueur du texte, nb de tokens, etc."
    )
    fe_hints.append("use_text_embeddings")
    fe_hints.append("tfidf_or_ngram_features")

    if missing_rate > 0:
        recommendations.append(
            "Remplacer les NaN par une chaîne spéciale (ex: '__MISSING_TEXT__')."
        )
        fe_hints.append("text_missing_token")

    text_stats = _build_text_stats(
        series,
        n_examples=getattr(config, "example_values_per_col", 5),
    )

    return notes, recommendations, fe_hints, text_stats


def _recommendations_datetime(
    notes: list[str],
    recommendations: list[str],
    fe_hints: list[str],
    missing_rate: float,
) -> tuple[list[str], list[str], list[str]]:
    """Génère recommandations pour une feature datetime."""
    notes.append("Colonne de type date/temps.")
    recommendations.append(
        "Extraire des features temporelles : année, mois, jour, jour de semaine, "
        "heure, indicateur week-end, écarts de temps par rapport à un événement, etc."
    )
    fe_hints.append("extract_datetime_features")

    if missing_rate > 0:
        recommendations.append(
            "Imputer les dates manquantes ou les traiter comme événement 'inconnu'."
        )
        fe_hints.append("datetime_imputation")

    return notes, recommendations, fe_hints


def _recommendations_boolean(
    notes: list[str],
    recommendations: list[str],
    fe_hints: list[str],
    missing_rate: float,
) -> tuple[list[str], list[str], list[str]]:
    """Génère recommandations pour une feature booléenne."""
    recommendations.append("Encodage 0/1 ou True/False -> 1/0.")
    fe_hints.append("binary_feature")

    if missing_rate > 0:
        recommendations.append(
            "Imputer les valeurs manquantes par la modalité dominante ou une 3ᵉ catégorie."
        )
        fe_hints.append("boolean_imputation")

    return notes, recommendations, fe_hints


def _determine_llm_type(
    role: str,
    high_cardinality: bool,
    is_constant: bool,
    is_id_like: bool,
) -> tuple[str, str]:
    """
    Détermine le rôle et type pour le LLM.

    Returns:
        Tuple[llm_role, inferred_type]
    """
    # Rôle LLM
    if role == "text":
        llm_role = "text"
    elif role == "datetime":
        llm_role = "timestamp"
    else:
        llm_role = "feature"

    # Type inféré
    if role == "numeric":
        inferred_type = "numeric"
    elif role == "categorical":
        inferred_type = "categorical_high" if high_cardinality else "categorical_low"
    elif role == "text":
        inferred_type = "text"
    elif role == "datetime":
        inferred_type = "datetime"
    elif role == "boolean":
        inferred_type = "bool"
    else:
        if is_constant:
            inferred_type = "constant"
        elif is_id_like:
            inferred_type = "id_like"
        else:
            inferred_type = "unknown"

    return llm_role, inferred_type


def _build_feature_summary(
    col: str,
    series: pd.Series,
    role: str,
    dtype: str,
    n_rows: int,
    n_unique: int,
    unique_ratio: float,
    missing_rate: float,
    flags: list[str],
    notes: list[str],
    fe_hints: list[str],
    llm_role: str,
    inferred_type: str,
    numeric_stats: NumericStats | None,
    categorical_stats: CategoricalStats | None,
    text_stats: TextStats | None,
    config: FEAnalysisConfig,
) -> FeatureSummaryForLLM:
    """Construit la dataclass FeatureSummaryForLLM."""
    n_examples = getattr(config, "example_values_per_col", 5)

    example_values = series.dropna().astype(str).drop_duplicates().head(n_examples).tolist()

    return FeatureSummaryForLLM(
        name=col,
        role=llm_role,
        inferred_type=inferred_type,
        pandas_dtype=dtype,
        n_rows=n_rows,
        n_non_null=int(series.notna().sum()),
        n_missing=int(series.isna().sum()),
        missing_rate=missing_rate,
        n_unique=int(n_unique),
        unique_ratio=unique_ratio,
        example_values=example_values,
        numeric_stats=numeric_stats.__dict__ if numeric_stats else None,
        categorical_stats=categorical_stats.__dict__ if categorical_stats else None,
        text_stats=text_stats.__dict__ if text_stats else None,
        flags=flags,
        notes=list(notes),
        fe_hints=fe_hints,
        feature_description=None,
    )


def _collect_warnings(
    col: str,
    role: str,
    is_constant: bool,
    is_id_like: bool,
    high_cardinality: bool,
    n_unique: int,
) -> list[str]:
    """Collecte les warnings pour une feature."""
    warnings: list[str] = []

    if is_id_like:
        warnings.append(f"[{col}] ressemble à un identifiant (id-like).")
    if is_constant:
        warnings.append(f"[{col}] est constante (aucune variance).")
    if high_cardinality and role == "categorical":
        warnings.append(
            f"[{col}] est une catégorielle de haute cardinalité ({n_unique} modalités)."
        )

    return warnings


# ----------------------------------------------------------------------
# Fonction principale (refactorisée)
# ----------------------------------------------------------------------


def analyze_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: FEAnalysisConfig,
) -> dict[str, Any]:
    """
    Analyse les colonnes de features d'un DataFrame pour guider le feature engineering.

    Args:
        df: DataFrame à analyser
        feature_cols: Liste des colonnes features à analyser
        config: Configuration de l'analyse

    Returns:
        Dict avec:
        - "features": dict legacy pour rapport humain
        - "warnings": liste des warnings
        - "llm_features": dict {col: FeatureSummaryForLLM}
    """
    features_info: dict[str, Any] = {}
    all_warnings: list[str] = []
    llm_features: dict[str, FeatureSummaryForLLM] = {}

    n_rows = len(df)

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature '{col}' absente des colonnes du DataFrame.")

        series = df[col]
        dtype = str(series.dtype)
        n_unique = series.nunique(dropna=True)
        missing_rate = float(series.isna().mean())
        unique_ratio = n_unique / max(1, n_rows)

        logger.debug(f"Analyse de la feature '{col}' (dtype={dtype}, n_unique={n_unique})")

        # 1) Déterminer le rôle
        role, _ = _determine_role(series, n_unique, unique_ratio, config)

        # 2) Détecter les flags
        flags, is_constant, is_id_like, high_cardinality = _detect_flags(
            col, role, n_unique, unique_ratio, config
        )

        # 3) Générer recommandations et stats
        (
            notes,
            recommendations,
            fe_hints,
            extra_info,
            numeric_stats,
            categorical_stats,
            text_stats,
        ) = _generate_recommendations(
            series, role, missing_rate, is_constant, is_id_like, high_cardinality, n_unique, config
        )

        # 4) Déterminer type LLM
        llm_role, inferred_type = _determine_llm_type(
            role, high_cardinality, is_constant, is_id_like
        )

        # 5) Collecter warnings
        warnings = _collect_warnings(col, role, is_constant, is_id_like, high_cardinality, n_unique)
        all_warnings.extend(warnings)

        # 6) Construire dict legacy (rapport humain)
        features_info[col] = {
            "dtype": dtype,
            "role": role,
            "n_unique": int(n_unique),
            "unique_ratio": unique_ratio,
            "missing_rate": missing_rate,
            "is_constant": is_constant,
            "is_id_like": is_id_like,
            "high_cardinality": high_cardinality,
            "notes": notes,
            "recommendations": recommendations,
            "extra_info": extra_info,
        }

        # 7) Construire dataclass LLM
        llm_features[col] = _build_feature_summary(
            col=col,
            series=series,
            role=role,
            dtype=dtype,
            n_rows=n_rows,
            n_unique=n_unique,
            unique_ratio=unique_ratio,
            missing_rate=missing_rate,
            flags=flags,
            notes=notes,
            fe_hints=fe_hints,
            llm_role=llm_role,
            inferred_type=inferred_type,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            text_stats=text_stats,
            config=config,
        )

    return {
        "features": features_info,
        "warnings": all_warnings,
        "llm_features": llm_features,
    }
