from __future__ import annotations
from dataclasses import is_dataclass, asdict
from typing import Any, Dict
import numpy as np
import pandas as pd


def make_json_safe(obj: Any) -> Any:
    """
    Convertit récursivement un objet Python (dataclass, DataFrame, Series, numpy, etc.)
    en structure sérialisable en JSON.
    """

    # Déjà sérialisables
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Dataclass -> dict nettoyé
    if is_dataclass(obj):
        return make_json_safe(asdict(obj))

    # Numpy scalaires -> types Python
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    # Dict -> nettoyage récursif
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # List / tuple / set -> liste nettoyée
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    # DataFrame -> dict lisible
    if isinstance(obj, pd.DataFrame):
        if len(obj) <= 50:
            rows: Dict[str, Dict[str, Any]] = {
                str(idx): {col: make_json_safe(row[col]) for col in obj.columns}
                for idx, row in obj.iterrows()
            }
            return {"__type__": "DataFrame", "rows": rows}

        head = obj.head(5)
        rows: Dict[str, Dict[str, Any]] = {
            str(idx): {col: make_json_safe(row[col]) for col in head.columns}
            for idx, row in head.iterrows()
        }
        return {
            "__type__": "DataFrame",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "head_rows": rows,
        }

    # Series -> résumé
    if isinstance(obj, pd.Series):
        return {
            "__type__": "Series",
            "size": int(obj.size),
            "values": [make_json_safe(v) for v in obj.head(20).tolist()],
        }

    # Fallback : serialize as string
    return str(obj)
