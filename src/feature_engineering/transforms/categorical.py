# categorical_transforms.py
import pandas as pd
from src.features_engineering.transformation_fe.registry import register


@register("one_hot")
def one_hot(col):
    if isinstance(col, pd.Series):
        prefix = col.name if col.name else "feature"
        return pd.get_dummies(col, prefix=prefix)
    elif isinstance(col, str):
        # fallback : on crée un DataFrame avec une seule colonne
        return pd.get_dummies(pd.Series([col]*1), prefix="feature")
    else:
        raise TypeError(f"Cannot one_hot type {type(col)}")

@register("ordinal")
def ordinal(col):
    return col.astype("category").cat.codes

@register("reduce_cardinality_first_letter")
def reduce_cardinality_first_letter(col):
    return col.astype(str).str[0]


@register("target_encoding")
def target_encoding(
    col,
    y,
    out_of_fold,
    smoothing,
    n_folds = 5,
    strategy="mean",
    random_state=42
):
    """
    Target encoding robuste compatible avec la syntaxe LLM :

        target_encoding(Pclass, out_of_fold=True, smoothing=0.1)

    Paramètres :
    - col : pd.Series catégorielle
    - y   : pd.Series cible (OBLIGATOIRE)
    - out_of_fold : si True → évite la fuite de cible (méthode Kaggle)
    - smoothing : alpha du lissage (0 = pas de lissage, ex. 0.1 ou 0.3 recommandé)
    - strategy : "mean" ou "median"
    - n_folds : utilisé uniquement si out_of_fold=True
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold

    if not isinstance(col, pd.Series):
        raise TypeError(f"target_encoding requires a pandas Series, got {type(col)}")

    if y is None:
        raise ValueError("target_encoding requires a target vector y")

    if len(col) != len(y):
        raise ValueError("col and y must have the same length")

    df = pd.DataFrame({"col": col, "y": y})

    # --------------------------
    # Fallback global
    # --------------------------
    if strategy == "mean":
        global_value = y.mean()
    elif strategy == "median":
        global_value = y.median()
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    # --------------------------
    # CAS SIMPLIFIÉ (pas OOF)
    # --------------------------
    if not out_of_fold:
        if strategy == "mean":
            stats = df.groupby("col")["y"].mean()
        else:
            stats = df.groupby("col")["y"].median()

        # Lissage façon CatBoost / Kaggle
        counts = df.groupby("col")["y"].count()
        smooth_stats = (counts * stats + smoothing * global_value) / (counts + smoothing)

        encoded = df["col"].map(smooth_stats).fillna(global_value)
        return encoded

    # --------------------------
    # CAS OOF (méthode Kaggle)
    # --------------------------
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_encoded = pd.Series(index=df.index, dtype=float)

    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Stats sur le fold d'entraînement seulement (évite fuite de cible)
        if strategy == "mean":
            stats = train_df.groupby("col")["y"].mean()
        else:
            stats = train_df.groupby("col")["y"].median()

        counts = train_df.groupby("col")["y"].count()

        smooth_stats = (counts * stats + smoothing * global_value) / (counts + smoothing)

        # Application au fold de validation
        encoded_val = val_df["col"].map(smooth_stats).fillna(global_value)
        oof_encoded.iloc[val_idx] = encoded_val

    return oof_encoded




@register("hashing")
def hashing(col, n_components):
    """
    Hashing trick simple pour colonnes catégorielles.
    Génère n_components colonnes numériques.
    """
    if not isinstance(col, pd.Series):
        raise TypeError(f"hashing() requires a pandas Series, got {type(col)}")

    n_components = int(n_components)

    # Hash pour chaque valeur
    hashed = col.astype(str).apply(lambda x: hash(x) % n_components)

    # Création d'un DataFrame dense one-hot des buckets
    df = pd.get_dummies(hashed)
    
    # S'assurer que toutes les colonnes existent
    for i in range(n_components):
        if i not in df.columns:
            df[i] = 0

    # Trier pour garantir un ordre stable
    df = df[sorted(df.columns)]

    # Renommer colonnes
    df.columns = [f"hash_{col.name}_{i}" for i in range(n_components)]

    return df


