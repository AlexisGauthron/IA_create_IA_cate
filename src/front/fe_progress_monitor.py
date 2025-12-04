"""
Moniteur de progression pour le Feature Engineering.
Lit les fichiers JSON générés par LLMFE pour afficher la progression en temps réel.
"""

import json
from pathlib import Path

import pandas as pd


def get_current_progress(samples_dir: Path) -> pd.DataFrame:
    """
    Lit tous les fichiers JSON de samples et retourne un DataFrame avec la progression.

    Args:
        samples_dir: Chemin vers le dossier contenant les fichiers sample_*.json

    Returns:
        DataFrame avec colonnes: sample_order, score, best_cumulative
    """
    if not samples_dir.exists():
        return pd.DataFrame(columns=["sample_order", "score", "best_cumulative"])

    records = []

    # Lire tous les fichiers sample_*.json
    for f in samples_dir.glob("sample_*.json"):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                records.append(
                    {
                        "sample_order": data.get("sample_order", 0),
                        "score": data.get("score"),
                        "function": data.get("function", ""),
                    }
                )
        except (OSError, json.JSONDecodeError):
            # Fichier peut être en cours d'écriture, on l'ignore
            continue

    if not records:
        return pd.DataFrame(columns=["sample_order", "score", "best_cumulative"])

    df = pd.DataFrame(records)
    df = df.sort_values("sample_order").reset_index(drop=True)

    # Calculer le meilleur score cumulé
    best_so_far = []
    current_best = float("-inf")
    for score in df["score"]:
        if score is not None and score > current_best:
            current_best = score
        best_so_far.append(current_best if current_best > float("-inf") else None)

    df["best_cumulative"] = best_so_far

    return df


def get_metrics(df: pd.DataFrame) -> dict[str, any]:
    """
    Calcule les métriques à partir du DataFrame de progression.

    Args:
        df: DataFrame retourné par get_current_progress()

    Returns:
        Dictionnaire avec les métriques
    """
    if len(df) == 0:
        return {
            "total": 0,
            "valid": 0,
            "failed": 0,
            "best_score": None,
            "last_score": None,
            "success_rate": 0,
        }

    valid_df = df[df["score"].notna()]
    failed_df = df[df["score"].isna()]

    best_score = valid_df["score"].max() if len(valid_df) > 0 else None
    last_score = df.iloc[-1]["score"] if len(df) > 0 else None

    return {
        "total": len(df),
        "valid": len(valid_df),
        "failed": len(failed_df),
        "best_score": best_score,
        "last_score": last_score,
        "success_rate": (len(valid_df) / len(df) * 100) if len(df) > 0 else 0,
    }


def get_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données pour le graphique Streamlit.

    Args:
        df: DataFrame retourné par get_current_progress()

    Returns:
        DataFrame prêt pour st.line_chart()
    """
    if len(df) == 0:
        return pd.DataFrame()

    # Ne garder que les colonnes utiles pour le graphique
    chart_df = df[["sample_order", "score", "best_cumulative"]].copy()

    # Renommer pour un affichage plus clair
    chart_df = chart_df.rename(columns={"score": "Score", "best_cumulative": "Meilleur cumulé"})

    # Utiliser sample_order comme index
    chart_df = chart_df.set_index("sample_order")

    return chart_df


def get_best_model(df: pd.DataFrame) -> dict | None:
    """
    Retourne les informations sur le meilleur modèle.

    Args:
        df: DataFrame retourné par get_current_progress()

    Returns:
        Dictionnaire avec sample_order, score, function ou None
    """
    if len(df) == 0:
        return None

    valid_df = df[df["score"].notna()]
    if len(valid_df) == 0:
        return None

    best_idx = valid_df["score"].idxmax()
    best_row = df.loc[best_idx]

    return {
        "sample_order": best_row["sample_order"],
        "score": best_row["score"],
        "function": best_row.get("function", ""),
    }


def get_recent_samples(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """
    Retourne les N derniers samples.

    Args:
        df: DataFrame retourné par get_current_progress()
        n: Nombre de samples à retourner

    Returns:
        Liste de dictionnaires avec les infos des samples récents
    """
    if len(df) == 0:
        return []

    recent = df.tail(n).iloc[::-1]  # Inverser pour avoir le plus récent en premier

    return [
        {
            "sample_order": row["sample_order"],
            "score": row["score"],
            "status": "✅" if row["score"] is not None else "❌",
        }
        for _, row in recent.iterrows()
    ]


def load_final_summary(results_dir: Path) -> dict | None:
    """
    Charge le résumé final si disponible.

    Args:
        results_dir: Chemin vers le dossier results/

    Returns:
        Dictionnaire avec le résumé ou None
    """
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_best_model_json(results_dir: Path) -> dict | None:
    """
    Charge le meilleur modèle depuis le JSON final.

    Args:
        results_dir: Chemin vers le dossier results/

    Returns:
        Dictionnaire avec le meilleur modèle ou None
    """
    best_path = results_dir / "best_model.json"
    if not best_path.exists():
        return None

    try:
        with open(best_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
