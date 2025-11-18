from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def df_to_list(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare un DataFrame en X_train, X_test, y_train, y_test.

    Paramètres
    ----------
    df : pd.DataFrame
        Jeu de données complet (features + cible).
    target_col : str
        Nom de la colonne cible.
    test_size : float
        Proportion de l'échantillon de test (0 < test_size < 1).
    random_state : int
        Graine aléatoire pour la reproductibilité.
    stratify : bool
        Si True, essaie une séparation stratifiée (utile en classification).

    Retour
    ------
    (X_train, X_test, y_train, y_test)
    """

    if target_col not in df.columns:
        raise KeyError(
            f"'{target_col}' n'est pas une colonne du DataFrame. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # On ne stratifie que si pertinent et possible (pas de NaN, au moins 2 classes)
    stratify_arg = None
    if stratify and y.nunique(dropna=False) > 1 and y.isna().sum() == 0:
        # vérification rapide pour éviter les erreurs quand certaines classes sont trop rares
        vc_min = y.value_counts().min()
        stratify_arg = y if vc_min >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    return X_train, X_test, y_train, y_test



def df_to_list_Kaggle(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "target",
    align_columns: bool = True,
    fill_missing: float | int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    À partir de df_train et df_test déjà séparés, retourne X_train, X_test, y_train, y_test.

    Paramètres
    ----------
    df_train : pd.DataFrame
        DataFrame d'entraînement contenant la colonne cible.
    df_test : pd.DataFrame
        DataFrame de test contenant la colonne cible.
    target_col : str
        Nom de la colonne cible.
    align_columns : bool
        Si True, réindexe X_test pour avoir exactement les colonnes de X_train (ordre identique).
    fill_missing : float | int
        Valeur de remplissage pour les colonnes manquantes dans X_test lors de l'alignement.

    Retour
    ------
    (X_train, X_test, y_train, y_test)
    """

    # Vérifications de base
    for name, df in [("df_train", df_train), ("df_test", df_test)]:
        if target_col not in df.columns:
            raise KeyError(f"'{target_col}' n'est pas une colonne de {name}. Colonnes: {list(df.columns)}")

    # Séparation features / cible
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test
 
    # Harmonisation des colonnes (facultative mais pratique)
    if align_columns:
        # Conserver uniquement l'ordre/ensemble des colonnes d'entraînement
        X_test = X_test.reindex(columns=X_train.columns, fill_value=fill_missing)

    return X_train, X_test, y_train
