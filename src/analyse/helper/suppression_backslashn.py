from __future__ import annotations

from typing import Any


def normalize_string_whitespace(obj: Any) -> Any:
    """
    Parcourt récursivement une structure JSON-like (dict / list / scalaires)
    et remplace les retours à la ligne dans les chaînes par des espaces simples.
    """
    if isinstance(obj, str):
        # remplace les vrais retours à la ligne par un espace
        # et évite les doubles espaces inutiles
        s = obj.replace("\r\n", " ").replace("\n", " ")
        s = " ".join(s.split())
        return s

    if isinstance(obj, dict):
        return {k: normalize_string_whitespace(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [normalize_string_whitespace(v) for v in obj]

    return obj
