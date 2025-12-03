import pandas as pd

def get_unique_columns_dataframe(df1: pd.DataFrame, df2: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Retourne un DataFrame contenant uniquement les colonnes différentes
    entre df1 et df2 (différence symétrique), tout en conservant la colonne cible.
    
    Hypothèse : df1 et df2 ont le même nombre de lignes.
    Aucun NaN n'est ajouté.
    """

    # Colonnes hors cible
    cols1 = set(df1.columns) - {target_col}
    cols2 = set(df2.columns) - {target_col}

    # Colonnes uniques dans l'un ou l'autre
    unique_cols = cols1.symmetric_difference(cols2)

    print("\n\n\nUNIAUYEEHEHEIUAGEG",unique_cols)
    # Ajouter la colonne cible
    unique_cols.add(target_col)

    # Sélectionner uniquement les colonnes existantes dans df1
    df_result = df2[[col for col in unique_cols if col in df2.columns]]

    return df_result





def drop_columns(df: pd.DataFrame, columns_to_drop: list):
    """
    Retourne un nouveau DataFrame sans les colonnes spécifiées dans 'columns_to_drop'.
    Ignore automatiquement les colonnes inexistantes.
    """
    return df.drop(columns=columns_to_drop, errors='ignore').copy()


def count_features(df: pd.DataFrame) -> int:
    """
    Retourne le nombre total de features (colonnes) dans un DataFrame.
    """
    return df.shape[1]
