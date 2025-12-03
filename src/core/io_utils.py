from pathlib import Path
from typing import Tuple, Optional, Union
import pandas as pd
import csv
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits


# ============== Fonctions de chargement de datasets sklearn ==============

def load_datasets_breast_cancer() -> pd.DataFrame:
    """Charge le dataset breast_cancer de sklearn."""
    print("[INFO] Load datasets breast_cancer\n")
    ds = load_breast_cancer(as_frame=True)
    return ds.frame

def load_datasets_iris() -> pd.DataFrame:
    """Charge le dataset iris de sklearn."""
    print("[INFO] Load datasets iris\n")
    ds = load_iris(as_frame=True)
    return ds.frame

def load_datasets_wine() -> pd.DataFrame:
    """Charge le dataset wine de sklearn."""
    print("[INFO] Load datasets wine\n")
    ds = load_wine(as_frame=True)
    return ds.frame

def load_datasets_digits() -> pd.DataFrame:
    """Charge le dataset digits de sklearn."""
    print("[INFO] Load datasets digits\n")
    ds = load_digits(as_frame=True)
    return ds.frame


# ============== Fonction de chargement train/test depuis CSV ==============

def csv_to_dataframe_train_test(
    nom_dossier: Union[str, Path],
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    train_pattern: str = "train",
    test_pattern: str = "test",
    **read_csv_kwargs,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Charge les fichiers CSV contenant "train" et (optionnellement) "test" dans leurs noms.

    Parametres:
      - nom_dossier: dossier a parcourir.
      - sep, encoding: parametres transmis a pandas.
      - train_pattern, test_pattern: motifs recherches dans les noms de fichiers.
      - read_csv_kwargs: parametres additionnels transmis a pandas.read_csv.

    Retour:
      - tuple (train_df, test_df ou None si le fichier test n'existe pas).
    """
    dossier = Path(nom_dossier)
    if not dossier.exists():
        raise FileNotFoundError(f"[ERROR] Le dossier '{nom_dossier}' est introuvable.")

    csv_files = list(dossier.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"[ERROR] Aucun fichier CSV trouve dans '{nom_dossier}'.")

    def _find_file(pattern: str, *, required: bool = True) -> Optional[Path]:
        matches = [
            fichier
            for fichier in csv_files
            if pattern.lower() in fichier.stem.lower()
        ]
        if not matches:
            if required:
                raise FileNotFoundError(
                    f"[ERROR] Aucun fichier CSV correspondant au motif '{pattern}' dans '{nom_dossier}'."
                )
            else:
                print(
                    f"[WARN] Aucun fichier CSV correspondant au motif '{pattern}' dans '{nom_dossier}'. "
                    f"Le fichier est ignore."
                )
                return None

        if len(matches) > 1:
            raise ValueError(
                f"[ERROR] Plusieurs fichiers CSV correspondent au motif '{pattern}' dans '{nom_dossier}'."
            )
        return matches[0]

    train_file = _find_file(train_pattern, required=True)
    test_file = _find_file(test_pattern, required=False)

    train_df = pd.read_csv(train_file, sep=sep, encoding=encoding, **read_csv_kwargs)
    print(f"[INFO] Train CSV charge: {train_df.shape[0]} lignes, {train_df.shape[1]} colonnes")

    test_df = None
    if test_file is not None:
        test_df = pd.read_csv(test_file, sep=sep, encoding=encoding, **read_csv_kwargs)
        print(f"[INFO] Test CSV charge: {test_df.shape[0]} lignes, {test_df.shape[1]} colonnes")

    return train_df, test_df


# ============== Fonctions de sauvegarde/lecture CSV ==============

def to_csv(
    df: pd.DataFrame,
    nom_fichier: str,
    nom_dossier: str = "Data",
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    index: bool = False,
    mode: str = "w",          # "w" écrase, "x" interdit d'écraser, "a" ajoute à la fin
    header: bool = True,
    na_rep: str = "",
    quoting: int = csv.QUOTE_MINIMAL
) -> Path:
    """
    Sauvegarde un DataFrame dans un fichier CSV.

    Paramètres:
      - df: DataFrame à écrire
      - path: chemin du fichier (dossiers créés si absents). L'extension .csv est ajoutée si manquante.
      - sep: séparateur de colonnes (par défaut ",")
      - encoding: encodage (par défaut "utf-8")
      - index: écrire l'index pandas (False par défaut)
      - mode: mode d'ouverture ("w" écrase, "x" échoue si le fichier existe, "a" ajoute)
      - header: écrire l’entête des colonnes (True par défaut)
      - na_rep: représentation des valeurs manquantes
      - quoting: stratégie de quoting (csv.QUOTE_MINIMAL par défaut)

    Retour:
      - Path vers le fichier écrit.
    """


    if not isinstance(df, pd.DataFrame):
        raise TypeError("[ERROR] df doit être un pandas.DataFrame\n")

    path = f"{nom_dossier}/{Path(nom_fichier).stem}"
    p = Path(path)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")

    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        p,
        sep=sep,
        encoding=encoding,
        index=index,
        mode=mode,
        header=header,
        na_rep=na_rep,
        quoting=quoting,
        lineterminator="\n",
    )
    return p


import pandas as pd
from typing import Optional

def to_dataframe(nom_fichier: str, 
                 nom_dossier: str = "Data",
                    index_col: Optional[str] = None, 
                    parse_dates: Optional[list] = None,
                    encoding: str = "utf-8") -> pd.DataFrame:
    """
    Lit un CSV et renvoie un DataFrame pandas.

    Paramètres :
    - file_path : chemin vers le fichier CSV
    - index_col : nom de la colonne à utiliser comme index (optionnel)
    - parse_dates : liste de colonnes à parser en datetime (optionnel)
    - encoding : encodage du fichier CSV (default='utf-8')

    Retour :
    - pd.DataFrame
    """
    try:
        df = pd.read_csv(f"{nom_dossier}/{nom_fichier}", index_col=index_col, parse_dates=parse_dates, encoding=encoding)
        print(f"[INFO] CSV chargé avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier non trouvé : {nom_fichier}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Erreur parsing CSV : {e}")
    except Exception as e:
        raise RuntimeError(f"Erreur inconnue lors de la lecture du CSV : {e}")

