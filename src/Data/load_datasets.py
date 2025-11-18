import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path
from typing import Tuple, Union
import pandas as pd


from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits



def load_datasets_breast_cancer():
    print("[INFO] Load datasets\n")
    ds = load_breast_cancer(as_frame=True) 
    df = ds.frame   
    return df

def load_datasets_iris():
    print("[INFO] Load datasets\n")
    ds = load_iris(as_frame=True)
    df = ds.frame
    return df

def load_datasets_wine():
    print("[INFO] Load datasets\n")
    ds = load_wine(as_frame=True)
    df = ds.frame 
    return df

def load_datasets_digits():
    print("[INFO] Load datasets\n")
    ds = load_digits(as_frame=True)
    df = ds.frame 
    return df



from typing import Union, Tuple, Optional
from pathlib import Path
import pandas as pd


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
      - train_pattern, test_pattern: motifs recherches dans les noms de fichiers sans leur extension.
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

    # Train obligatoire
    train_path = _find_file(train_pattern, required=True)
    train_df = pd.read_csv(train_path, sep=sep, encoding=encoding, **read_csv_kwargs)

    # Test optionnel
    test_path = _find_file(test_pattern, required=False)
    if test_path is not None:
        test_df = pd.read_csv(test_path, sep=sep, encoding=encoding, **read_csv_kwargs)
    else:
        test_df = None

    return train_df, test_df
