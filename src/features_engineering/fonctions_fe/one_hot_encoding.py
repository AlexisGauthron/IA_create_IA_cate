import pandas as pd
from typing import List, Tuple

def one_hot_encode(
    df: pd.DataFrame,
    categorical_cols: List[str],
    drop_first: bool = False,
    dtype: str = "int8"
) -> pd.DataFrame:
    """
    Applique un one-hot encoding sur les colonnes catégorielles spécifiées.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée (non modifié en place).
    categorical_cols : List[str]
        Liste des noms de colonnes à encoder.
    drop_first : bool, par défaut False
        Si True, on supprime la première modalité (évite colinéarité parfaite).
    dtype : str, par défaut "int8"
        Type des colonnes encodées (int8 permet de gagner de la mémoire).

    Retour
    ------
    pd.DataFrame
        Nouveau DataFrame avec les colonnes encodées.
    """
    # On travaille sur une copie pour ne pas modifier df original
    df_copy = df.copy()

    # Vérifier que les colonnes existent
    missing = [c for c in categorical_cols if c not in df_copy.columns]
    if missing:
        raise ValueError(f"Colonnes introuvables dans le DataFrame : {missing}")

    # One-hot encoding via pandas
    df_encoded = pd.get_dummies(
        df_copy,
        columns=categorical_cols,
        drop_first=drop_first,
        dtype=dtype
    )

    return df_encoded
