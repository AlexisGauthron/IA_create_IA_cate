from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ----------------------------
# Stats classes
# ----------------------------
from src.analyse.dataset.type_stats import CategoricalStats, NumericStats, TextStats


@dataclass
class FeatureSummaryForLLM:
    """
    Snapshot d'une colonne (feature ou autre) pour guider un LLM
    dans le choix des transformations de feature engineering.
    """

    name: str

    # Rôle logique dans le pipeline
    role: Literal[
        "feature",  # X standard
        "target",  # y
        "id",  # identifiant pur
        "timestamp",  # temps / date
        "group",  # id client, id session, etc.
        "text",  # texte libre (verbatim)
    ] = "feature"

    # Type de données inféré (basé sur pandas + heuristiques)
    inferred_type: Literal[
        "numeric",
        "categorical_low",
        "categorical_high",
        "text",
        "datetime",
        "bool",
        "constant",
        "id_like",
        "unknown",
    ] = "unknown"

    pandas_dtype: str = "object"

    # Stats générales
    n_rows: int = 0
    n_non_null: int = 0
    n_missing: int = 0
    missing_rate: float = 0.0

    # Cardinalité brute
    n_unique: int = 0
    unique_ratio: float = 0.0  # n_unique / n_rows

    # Exemples bruts (pour le LLM)
    example_values: list[str] = field(default_factory=list)

    # Stats spécialisées (mutuellement exclusives selon type)
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    text_stats: TextStats | None = None

    # Flags / heuristiques
    flags: list[str] = field(default_factory=list)
    # ex: ["ID_LIKE", "HIGH_CARDINALITY", "CONSTANT", "SKEWED_RIGHT"]

    # Notes / avertissements “lisibles LLM”
    notes: list[str] = field(default_factory=list)
    # ex: ["Probablement un identifiant, à exclure comme feature brute."]

    # Hints de FE (sans décider à la place du LLM)
    fe_hints: list[str] = field(default_factory=list)
    # ex: ["candidate_for_target_encoding", "good_for_one_hot", "use_text_embeddings"]

    # Description métier facultative (important pour le LLM)
    feature_description: str | None = None
    # ex: "Texte libre saisi par le client décrivant le problème."
