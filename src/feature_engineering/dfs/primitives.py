# src/feature_engineering/dfs/primitives.py
"""
Définition des primitives DFS disponibles.

Les primitives sont les opérations de base utilisées par DFS pour créer des features.
- Aggregation primitives: Agrègent plusieurs lignes en une valeur (mean, sum, count...)
- Transform primitives: Transforment une valeur en une autre (year, month, log...)

Référence: https://featuretools.alteryx.com/en/stable/getting_started/primitives.html
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# AGGREGATION PRIMITIVES
# Utilisées pour agréger des valeurs de tables liées (relations one-to-many)
# =============================================================================

AGGREGATION_PRIMITIVES = {
    # --- Statistiques de base ---
    "count": {
        "description": "Compte le nombre de valeurs non-nulles",
        "category": "basic",
        "complexity": 1,
    },
    "sum": {
        "description": "Somme des valeurs",
        "category": "basic",
        "complexity": 1,
    },
    "mean": {
        "description": "Moyenne des valeurs",
        "category": "basic",
        "complexity": 1,
    },
    "median": {
        "description": "Médiane des valeurs",
        "category": "basic",
        "complexity": 1,
    },
    "mode": {
        "description": "Valeur la plus fréquente",
        "category": "basic",
        "complexity": 1,
    },
    "min": {
        "description": "Valeur minimale",
        "category": "basic",
        "complexity": 1,
    },
    "max": {
        "description": "Valeur maximale",
        "category": "basic",
        "complexity": 1,
    },
    "std": {
        "description": "Écart-type des valeurs",
        "category": "dispersion",
        "complexity": 2,
    },
    "skew": {
        "description": "Asymétrie de la distribution",
        "category": "dispersion",
        "complexity": 2,
    },
    # --- Comptages ---
    "num_unique": {
        "description": "Nombre de valeurs uniques",
        "category": "counting",
        "complexity": 1,
    },
    "percent_true": {
        "description": "Pourcentage de valeurs True",
        "category": "counting",
        "complexity": 1,
    },
    "num_true": {
        "description": "Nombre de valeurs True",
        "category": "counting",
        "complexity": 1,
    },
    # --- Temporel (si time_index) ---
    "time_since_last": {
        "description": "Temps depuis la dernière valeur",
        "category": "temporal",
        "complexity": 2,
    },
    "time_since_first": {
        "description": "Temps depuis la première valeur",
        "category": "temporal",
        "complexity": 2,
    },
    "trend": {
        "description": "Tendance linéaire des valeurs",
        "category": "temporal",
        "complexity": 3,
    },
    # --- Avancé ---
    "entropy": {
        "description": "Entropie de la distribution",
        "category": "advanced",
        "complexity": 3,
    },
    "n_most_common": {
        "description": "N valeurs les plus fréquentes",
        "category": "advanced",
        "complexity": 2,
    },
    "first": {
        "description": "Première valeur (chronologique)",
        "category": "temporal",
        "complexity": 1,
    },
    "last": {
        "description": "Dernière valeur (chronologique)",
        "category": "temporal",
        "complexity": 1,
    },
}

# =============================================================================
# TRANSFORM PRIMITIVES
# Utilisées pour transformer des valeurs individuelles
# =============================================================================

TRANSFORM_PRIMITIVES = {
    # --- Datetime ---
    "year": {
        "description": "Extrait l'année",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "month": {
        "description": "Extrait le mois",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "day": {
        "description": "Extrait le jour",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "weekday": {
        "description": "Jour de la semaine (0-6)",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "hour": {
        "description": "Extrait l'heure",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "minute": {
        "description": "Extrait les minutes",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "is_weekend": {
        "description": "True si weekend",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "week": {
        "description": "Numéro de semaine",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    "quarter": {
        "description": "Trimestre (1-4)",
        "category": "datetime",
        "input_type": "datetime",
        "complexity": 1,
    },
    # --- Numérique ---
    "absolute": {
        "description": "Valeur absolue",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 1,
    },
    "negate": {
        "description": "Valeur négative",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 1,
    },
    "percentile": {
        "description": "Percentile de la valeur",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 2,
    },
    "cum_sum": {
        "description": "Somme cumulative",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 2,
    },
    "cum_mean": {
        "description": "Moyenne cumulative",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 2,
    },
    "cum_min": {
        "description": "Minimum cumulatif",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 2,
    },
    "cum_max": {
        "description": "Maximum cumulatif",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 2,
    },
    "diff": {
        "description": "Différence avec la valeur précédente",
        "category": "numeric",
        "input_type": "numeric",
        "complexity": 2,
    },
    # --- Texte ---
    "num_characters": {
        "description": "Nombre de caractères",
        "category": "text",
        "input_type": "text",
        "complexity": 1,
    },
    "num_words": {
        "description": "Nombre de mots",
        "category": "text",
        "input_type": "text",
        "complexity": 1,
    },
    # --- Booléen ---
    "is_null": {
        "description": "True si valeur nulle",
        "category": "boolean",
        "input_type": "any",
        "complexity": 1,
    },
    "not": {
        "description": "Inverse booléen",
        "category": "boolean",
        "input_type": "boolean",
        "complexity": 1,
    },
    # --- Latitude/Longitude ---
    "latitude": {
        "description": "Extrait la latitude",
        "category": "geo",
        "input_type": "latlong",
        "complexity": 1,
    },
    "longitude": {
        "description": "Extrait la longitude",
        "category": "geo",
        "input_type": "latlong",
        "complexity": 1,
    },
}


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def get_primitives_for_task(
    task_type: str = "classification",
    has_datetime: bool = False,
    has_text: bool = False,
    complexity_level: int = 2,
) -> tuple[list[str], list[str]]:
    """
    Retourne les primitives recommandées selon le type de tâche.

    Args:
        task_type: 'classification' ou 'regression'
        has_datetime: True si le dataset contient des colonnes datetime
        has_text: True si le dataset contient des colonnes texte
        complexity_level: 1 (simple), 2 (standard), 3 (avancé)

    Returns:
        Tuple (agg_primitives, trans_primitives)
    """
    # Primitives d'agrégation selon la complexité
    agg_prims = []

    # Niveau 1: Basique
    agg_prims.extend(["count", "sum", "mean", "min", "max"])

    # Niveau 2: Standard
    if complexity_level >= 2:
        agg_prims.extend(["median", "std", "num_unique", "mode"])

    # Niveau 3: Avancé
    if complexity_level >= 3:
        agg_prims.extend(["skew", "entropy", "percent_true"])

    # Primitives temporelles si datetime
    if has_datetime and complexity_level >= 2:
        agg_prims.extend(["time_since_last", "first", "last"])
        if complexity_level >= 3:
            agg_prims.extend(["trend"])

    # Primitives de transformation selon la complexité
    trans_prims = []

    # Niveau 1: Basique
    trans_prims.extend(["is_null", "absolute"])

    # Datetime si présent
    if has_datetime:
        trans_prims.extend(["year", "month", "day", "weekday"])
        if complexity_level >= 2:
            trans_prims.extend(["hour", "is_weekend", "quarter", "week"])

    # Niveau 2: Standard
    if complexity_level >= 2:
        trans_prims.extend(["percentile"])

    # Niveau 3: Avancé
    if complexity_level >= 3:
        trans_prims.extend(["cum_sum", "cum_mean", "diff"])

    # Texte si présent
    if has_text:
        trans_prims.extend(["num_characters", "num_words"])

    return agg_prims, trans_prims


def get_primitives_by_category(category: str) -> dict[str, dict[str, Any]]:
    """
    Retourne toutes les primitives d'une catégorie.

    Args:
        category: 'basic', 'dispersion', 'counting', 'temporal', 'advanced',
                  'datetime', 'numeric', 'text', 'boolean', 'geo'

    Returns:
        Dictionnaire des primitives de cette catégorie
    """
    result = {}

    for name, info in AGGREGATION_PRIMITIVES.items():
        if info.get("category") == category:
            result[name] = info

    for name, info in TRANSFORM_PRIMITIVES.items():
        if info.get("category") == category:
            result[name] = info

    return result


def list_all_primitives() -> dict[str, list[str]]:
    """Liste toutes les primitives disponibles par type."""
    return {
        "aggregation": list(AGGREGATION_PRIMITIVES.keys()),
        "transform": list(TRANSFORM_PRIMITIVES.keys()),
    }
