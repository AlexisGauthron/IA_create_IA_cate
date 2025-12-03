from typing import List
import pandas as pd

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--"}

def _clean_labels(seq) -> List[str]:
    s = pd.Series(seq, dtype="object")

    # Supprime vrais NaN
    s = s[~s.isna()]

    # To string + trim
    s = s.astype(str).str.strip()

    # Supprime tokens "manquants" écrits en toutes lettres
    s = s[~s.str.lower().isin(MISSING_TOKENS)]

    # Dédoublonne en conservant l'ordre
    seen = set()
    out = []
    for x in s.tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out