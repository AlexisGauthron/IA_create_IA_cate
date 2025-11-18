from __future__ import annotations

from typing import Dict, Any, Sequence, List
import numpy as np
import pandas as pd

from .config import FEAnalysisConfig

# 👇 dataclasses pour le LLM
from src.analyse.dataset.features import (
    FeatureSummaryForLLM,
    NumericStats,
    CategoricalStats,
    TextStats,
)

from src.analyse.statistiques.features_types import _build_text_stats, _build_numeric_stats, _build_categorical_stats

# ----------------------------------------------------------------------
# Fonction principale
# ----------------------------------------------------------------------
def analyze_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: FEAnalysisConfig,
) -> Dict[str, Any]:
    """
    Analyse les colonnes de features d'un DataFrame pour guider le meilleur
    feature engineering possible.

    Returns
    -------
    result : Dict[str, Any]
        {
            "features": { ... },   # dict legacy pour ton rapport humain
            "warnings": [...],
            "llm_features": { nom_feature: FeatureSummaryForLLM, ... }
        }
    """
    features_info: Dict[str, Any] = {}
    warnings: List[str] = []
    llm_features: Dict[str, FeatureSummaryForLLM] = {}

    n_rows = len(df)

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature '{col}' absente des colonnes du DataFrame.")

        s = df[col]
        dtype = str(s.dtype)
        n_unique = s.nunique(dropna=True)
        print(f"Analyse de la feature '{col}' (dtype={dtype})...")
        print(f" - Nombre de valeurs uniques : {n_unique}")
        missing_rate = float(s.isna().mean())
        unique_ratio = n_unique / max(1, n_rows)

        # ------------------------------------------------------------------
        # 1) Détermination du rôle logique de la feature
        # ------------------------------------------------------------------
        role = "unknown"
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)
        is_bool_type = pd.api.types.is_bool_dtype(s)

        # bool-like par les valeurs (0/1, True/False)
        non_na_values = s.dropna().unique().tolist()
        is_bool_values = False
        if len(non_na_values) > 0:
            bool_like_set = {0, 1, True, False}
            try:
                is_bool_values = set(non_na_values).issubset(bool_like_set)
            except TypeError:
                # Si comparaison impossible (types non hashables), on ignore
                is_bool_values = False

        is_bool_like = is_bool_type or is_bool_values

        if pd.api.types.is_numeric_dtype(s) and not is_bool_like:
            role = "numeric"
        elif is_datetime:
            role = "datetime"
        elif is_bool_like:
            role = "boolean"
        elif pd.api.types.is_categorical_dtype(s.dtype) or s.dtype == "object":
            # Différencier texte libre vs catégorielle selon cardinalité et ratio d'unicité
            if (n_unique > config.max_unique_cat_low) and (
                unique_ratio > config.text_unique_ratio_threshold
            ):
                role = "text"
            else:
                role = "categorical"

        # ------------------------------------------------------------------
        # 2) Flags utiles pour le FE : constante, id-like, haute cardinalité
        # ------------------------------------------------------------------
        is_constant = (n_unique <= 1)

        is_id_like_name = any(key in col.lower() for key in ["id", "uuid", "guid"])
        is_id_like_ratio = unique_ratio > config.id_unique_ratio_threshold
        is_id_like = bool(is_id_like_name or is_id_like_ratio)

        high_cardinality = False
        if role == "categorical" and n_unique > config.high_cardinality_threshold:
            high_cardinality = True

        # Flags pour la dataclass LLM
        flags: List[str] = []
        if is_constant:
            flags.append("CONSTANT")
        if is_id_like:
            flags.append("ID_LIKE")
        if high_cardinality and role == "categorical":
            flags.append("HIGH_CARDINALITY")

        # ------------------------------------------------------------------
        # 3) Notes & recommandations FE
        # ------------------------------------------------------------------
        notes: List[str] = []
        recommendations: List[str] = []
        extra_info: Any = None
        fe_hints: List[str] = []  # hints “symboliques” pour le LLM

        # Constante
        if is_constant:
            notes.append("Variable constante : probablement à supprimer.")
            recommendations.append("Supprimer : aucune information prédictive.")
            fe_hints.append("drop_constant_feature")

        # ID-like
        if is_id_like:
            notes.append("Variable probablement identifiant (ID-like).")
            recommendations.append(
                "Ne pas utiliser telle quelle comme feature : risque de fuite ou de sur-apprentissage."
            )
            fe_hints.append("do_not_use_as_raw_feature")

        # Valeurs manquantes
        if missing_rate > 0:
            notes.append(f"Valeurs manquantes : {missing_rate:.1%}.")
            if missing_rate > config.high_missing_threshold:
                notes.append(
                    f"Taux de NaN élevé (> {config.high_missing_threshold:.0%}) : "
                    "envisager imputation robuste ou exclusion."
                )
                fe_hints.append("high_missing_rate")

        # --- stats spécifiques pour la dataclass
        numeric_stats: NumericStats | None = None
        categorical_stats: CategoricalStats | None = None
        text_stats: TextStats | None = None

        # ------------------------------------------------------------------
        # 3bis) Détail par rôle
        # ------------------------------------------------------------------
        # Rôle : numeric
        if role == "numeric":
            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            skew = s.dropna().skew() if s.notna().any() else np.nan
            extra_info = {
                "describe": desc,
                "skew": skew,
            }

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
            recommendations.append(
                "Winsorisation ou clipping possible pour limiter l’impact des outliers."
            )
            fe_hints.append("candidate_for_scaling")

            # 👉 stats numériques pour la dataclass
            numeric_stats = _build_numeric_stats(s)

        # Rôle : catégorielle
        elif role == "categorical":
            if high_cardinality:
                notes.append(f"Haute cardinalité : {n_unique} modalités.")
                recommendations.append(
                    "Éviter le one-hot naïf. Préférer target encoding, leave-one-out, hashing, "
                    "embeddings ou regroupement des modalités rares."
                )
                fe_hints.append("use_target_encoding_or_hashing")
            else:
                recommendations.append(
                    "One-hot encoding ou encodage ordinal selon le modèle choisi."
                )
                fe_hints.append("candidate_for_one_hot")

            if missing_rate > 0:
                recommendations.append(
                    "Imputation par catégorie spéciale '__MISSING__' ou valeur NA explicite."
                )
                fe_hints.append("categorical_imputation")

            # 👉 stats catégorielles pour la dataclass
            categorical_stats = _build_categorical_stats(
                s,
                n_rows=n_rows,
                n_top=10,
                rare_level_threshold=getattr(config, "rare_level_threshold", 0.01),
            )

        # Rôle : texte
        elif role == "text":
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

            # 👉 stats texte pour la dataclass
            text_stats = _build_text_stats(
                s,
                n_examples=getattr(config, "example_values_per_col", 5),
            )

        # Rôle : datetime
        elif role == "datetime":
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

        # Rôle : booléen
        elif role == "boolean":
            recommendations.append("Encodage 0/1 ou True/False -> 1/0.")
            fe_hints.append("binary_feature")

            if missing_rate > 0:
                recommendations.append(
                    "Imputer les valeurs manquantes par la modalité dominante ou une 3ᵉ catégorie."
                )
                fe_hints.append("boolean_imputation")

        # Cas inconnu / ambigu
        else:
            recommendations.append(
                "Type non clairement identifié : vérifier la sémantique de cette colonne. "
                "Décider si elle doit être traitée comme numérique, catégorielle, texte, etc."
            )
            fe_hints.append("check_semantics")

        # ------------------------------------------------------------------
        # 4) Warnings globaux pour cette feature
        # ------------------------------------------------------------------
        if is_id_like:
            warnings.append(f"[{col}] ressemble à un identifiant (id-like).")
        if is_constant:
            warnings.append(f"[{col}] est constante (aucune variance).")
        if high_cardinality and role == "categorical":
            warnings.append(
                f"[{col}] est une catégorielle de haute cardinalité ({n_unique} modalités)."
            )

        # ------------------------------------------------------------------
        # 5) Dict “legacy” pour ton rapport humain
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 6) Dataclass FeatureSummaryForLLM pour le LLM
        # ------------------------------------------------------------------
        # rôle logique pour le LLM
        llm_role = "feature"
        if role == "text":
            llm_role = "text"
        elif role == "datetime":
            llm_role = "timestamp"

        # type inféré pour le LLM
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

        n_examples = getattr(config, "example_values_per_col", 5)

        example_values = (
            s.dropna()
            .astype(str)
            .drop_duplicates()   # 👈 garde seulement les valeurs uniques
            .head(n_examples)    # 👈 max n exemples
            .tolist()
        )

        llm_features[col] = FeatureSummaryForLLM(
            name=col,
            role=llm_role,
            inferred_type=inferred_type,
            pandas_dtype=dtype,
            n_rows=n_rows,
            n_non_null=int(s.notna().sum()),
            n_missing=int(s.isna().sum()),
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

    result: Dict[str, Any] = {
        "features": features_info,
        "warnings": warnings,
        "llm_features": llm_features,
    }
    return result
