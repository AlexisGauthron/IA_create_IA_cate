from __future__ import annotations
from typing import Dict, Any, List, Literal, Optional
from dataclasses import dataclass


FeatureType = Literal[
    "numeric_derived",
    "categorical_encoding",
    "datetime_derived",
    "text_representation",
]


@dataclass
class FEGenerationConfig:
    max_unary_numeric_per_col: int = 4
    max_pairwise_interactions: int = 50
    generate_family_size: bool = True   # ex: Titanic
    allow_text_embeddings: bool = True
    allow_text_tfidf: bool = False      # à activer si tu veux
    max_cat_for_one_hot: int = 10
    high_cardinality_threshold: int = 50


def generate_feature_space(
    report: Dict[str, Any],
    config: Optional[FEGenerationConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Génère un espace de features candidates sous forme de specs JSON-like.
    Ne calcule PAS les features, décrit juste les transformations.
    """
    if config is None:
        config = FEGenerationConfig()

    g = report["basic_stats"]
    n_rows = g["n_rows"]

    features = report["features"]

    candidates: List[Dict[str, Any]] = []

    # --- Helpers simples ---
    def infer_role(ft: Dict[str, Any]) -> str:
        t = ft.get("inferred_type", "unknown")
        if t in ("numeric", "integer", "float"):
            return "numeric"
        if t in ("categorical_low", "categorical_high", "categorical"):
            return "categorical"
        if t == "text":
            return "text"
        if t in ("datetime", "date", "timestamp"):
            return "datetime"
        if t in ("bool", "boolean"):
            return "boolean"
        return "unknown"

    # Liste des colonnes par rôle
    numeric_cols = []
    cat_cols = []
    text_cols = []
    datetime_cols = []

    name2info = {f["name"]: f for f in features}

    for f in features:
        role = infer_role(f)
        name = f["name"]
        if role == "numeric":
            numeric_cols.append(name)
        elif role == "categorical":
            cat_cols.append(name)
        elif role == "text":
            text_cols.append(name)
        elif role == "datetime":
            datetime_cols.append(name)

    # ============================================================
    # 1) NUMERIC – transformations unaires & interactions
    # ============================================================
    for col in numeric_cols:
        finfo = name2info[col]

        # Unaires principales
        # (tu peux ajouter / enlever selon ton goût)
        unary_transforms = [
            ("log1p", "log1p({col})"),
            ("sqrt", "sqrt({col})"),
            ("square", "({col} ** 2)"),
            ("cube", "({col} ** 3)"),
        ][: config.max_unary_numeric_per_col]

        for t_name, expr_tpl in unary_transforms:
            candidates.append(
                {
                    "name": f"{col}__{t_name}",
                    "type": "numeric_derived",
                    "inputs": [col],
                    "transformation": expr_tpl.format(col=col),
                    "reason": f"Transformation numérique uniaire {t_name} sur {col}.",
                }
            )

    # Interactions pairwise (simple ex, à limiter !)
    # ici on ne prend que les 50 premières pour éviter l'explosion
    pair_count = 0
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1 :]:
            if pair_count >= config.max_pairwise_interactions:
                break

            for op_name, expr_tpl in [
                ("prod", "({c1} * {c2})"),
                ("ratio", "({c1} / ({c2} + 1e-6))"),
                ("sum", "({c1} + {c2})"),
            ]:
                candidates.append(
                    {
                        "name": f"{c1}__{c2}__{op_name}",
                        "type": "numeric_derived",
                        "inputs": [c1, c2],
                        "transformation": expr_tpl.format(c1=c1, c2=c2),
                        "reason": f"Interaction {op_name} entre {c1} et {c2}.",
                    }
                )

            pair_count += 1
        if pair_count >= config.max_pairwise_interactions:
            break

    # Exemple métier Titanic : FamilySize
    if config.generate_family_size and {"SibSp", "Parch"}.issubset(set(numeric_cols)):
        candidates.append(
            {
                "name": "FamilySize",
                "type": "numeric_derived",
                "inputs": ["SibSp", "Parch"],
                "transformation": "FamilySize = SibSp + Parch + 1",
                "reason": "Taille de la famille, souvent corrélée à la survie.",
            }
        )

    # ============================================================
    # 2) CATEGORICAL – encodages candidates
    # ============================================================
    for col in cat_cols:
        finfo = name2info[col]
        n_unique = finfo.get("n_unique", None)
        flags = finfo.get("flags", []) or []

        # Si ID_LIKE -> on peut quand même proposer une feature dérivée type fréquence, sinon on saute
        if "ID_LIKE" in flags:
            candidates.append(
                {
                    "name": f"{col}__freq",
                    "type": "numeric_derived",
                    "inputs": [col],
                    "transformation": f"freq_encoding({col})",
                    "reason": (
                        f"{col} marqué ID_LIKE, on propose seulement un encodage "
                        "en fréquence plutôt qu'un one-hot brut."
                    ),
                }
            )
            continue

        # one-hot si faible cardinalité
        if n_unique is not None and n_unique <= config.max_cat_for_one_hot:
            candidates.append(
                {
                    "name": f"{col}__one_hot",
                    "type": "categorical_encoding",
                    "inputs": [col],
                    "encoding": "one_hot",
                    "transformation": f"one_hot({col})",
                    "reason": (
                        f"{col} catégorielle de faible cardinalité (n_unique={n_unique}), "
                        "one-hot adapté."
                    ),
                }
            )
        # target / hashing si plus élevé
        if n_unique is not None and n_unique >= config.high_cardinality_threshold:
            candidates.append(
                {
                    "name": f"{col}__target",
                    "type": "categorical_encoding",
                    "inputs": [col],
                    "encoding": "target_encoding",
                    "transformation": f"target_encoding({col})",
                    "reason": (
                        f"{col} haute cardinalité (n_unique={n_unique}), "
                        "target_encoding recommandé."
                    ),
                }
            )
            candidates.append(
                {
                    "name": f"{col}__hashing",
                    "type": "categorical_encoding",
                    "inputs": [col],
                    "encoding": "hashing",
                    "transformation": f"hashing({col}, n_components=k)",
                    "reason": (
                        f"{col} haute cardinalité (n_unique={n_unique}), "
                        "hashing possible pour dimension fixe."
                    ),
                }
            )

        # cardinalité intermédiaire -> laisser le choix (one_hot + target)
        if (
            n_unique is not None
            and config.max_cat_for_one_hot < n_unique < config.high_cardinality_threshold
        ):
            candidates.append(
                {
                    "name": f"{col}__one_hot",
                    "type": "categorical_encoding",
                    "inputs": [col],
                    "encoding": "one_hot",
                    "transformation": f"one_hot({col})",
                    "reason": (
                        f"{col} cardinalité intermédiaire (n_unique={n_unique}), "
                        "one_hot possible mais attention à la dimension."
                    ),
                }
            )
            candidates.append(
                {
                    "name": f"{col}__target",
                    "type": "categorical_encoding",
                    "inputs": [col],
                    "encoding": "target_encoding",
                    "transformation": f"target_encoding({col})",
                    "reason": (
                        f"{col} cardinalité intermédiaire (n_unique={n_unique}), "
                        "target_encoding souvent plus compact."
                    ),
                }
            )

    # ============================================================
    # 3) TEXT – candidats embeddings / tfidf
    # ============================================================
    for col in text_cols:
        if config.allow_text_embeddings:
            candidates.append(
                {
                    "name": f"{col}__embedding",
                    "type": "text_representation",
                    "inputs": [col],
                    "text_strategy": "embedding",
                    "model": "intfloat/multilingual-e5-base",
                    "reason": (
                        f"{col} est une feature texte, représentation par embeddings "
                        "pré-entraînés (E5)."
                    ),
                }
            )
        if config.allow_text_tfidf:
            candidates.append(
                {
                    "name": f"{col}__tfidf",
                    "type": "text_representation",
                    "inputs": [col],
                    "text_strategy": "tfidf",
                    "reason": f"{col} texte, représentation TF-IDF n-gram possible.",
                }
            )

    # ============================================================
    # 4) DATETIME – composants simples
    # ============================================================
    for col in datetime_cols:
        for comp in ["year", "month", "day", "dayofweek"]:
            candidates.append(
                {
                    "name": f"{col}__{comp}",
                    "type": "datetime_derived",
                    "inputs": [col],
                    "transformation": f"{comp}({col})",
                    "reason": f"Composant temporel {comp} dérivé de {col}.",
                }
            )

    return candidates
