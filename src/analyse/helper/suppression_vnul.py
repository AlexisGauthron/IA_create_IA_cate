from __future__ import annotations
from typing import Any, Dict, List

# clés pour lesquelles on accepte de garder la valeur None
_ALLOWED_NULL_KEYS = {
    "business_description",
    "metric",
    "feature_description",
}


def remove_nulls(obj: Any) -> Any:
    """
    Nettoyage récursif d'une structure JSON-like (dict / list / scalaires).

    - Dans un dict :
        * supprime les clés dont la valeur est None,
          SAUF si la clé est dans _ALLOWED_NULL_KEYS (on garde key: None),
        * supprime aussi les clés dont la valeur est une liste vide [] ou un dict vide {}.
    - Dans une liste :
        * supprime les éléments qui valent None,
        * supprime les éléments qui deviennent [] ou {} après nettoyage récursif.
    - Ne touche pas aux valeurs 0, False, "" etc.
    """
    # --- dict ---
    if isinstance(obj, dict):
        new_dict: Dict[Any, Any] = {}
        for k, v in obj.items():
            # Cas particulier : on autorise certains champs à rester None
            if v is None and k in _ALLOWED_NULL_KEYS:
                new_dict[k] = None
                continue

            cleaned = remove_nulls(v)

            # on supprime :
            #  - les None
            #  - les listes vides
            #  - les dicts vides
            if cleaned is None:
                continue
            if isinstance(cleaned, (list, dict)) and len(cleaned) == 0:
                continue

            new_dict[k] = cleaned

        return new_dict

    # --- list / tuple ---
    if isinstance(obj, (list, tuple)):
        new_list: List[Any] = []
        for v in obj:
            cleaned = remove_nulls(v)

            if cleaned is None:
                continue
            if isinstance(cleaned, (list, dict)) and len(cleaned) == 0:
                continue

            new_list.append(cleaned)

        return new_list

    # --- scalaires ---
    return obj
