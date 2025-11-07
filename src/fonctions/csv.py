from pathlib import Path
import pandas as pd
import csv

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
        line_terminator="\n",
    )
    return p



