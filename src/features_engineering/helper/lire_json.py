from __future__ import annotations
from typing import Any, Dict
import json
from pathlib import Path


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Lit un fichier JSON et renvoie son contenu sous forme de dict.

    Parameters
    ----------
    path : str | Path
        Chemin du fichier JSON.

    Returns
    -------
    data : dict
        Contenu du fichier JSON.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data
