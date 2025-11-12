from typing import Dict, List, Optional
import json, re, random
from collections import Counter


# ---------------- Helpers: nettoyage & tokenisation ----------------
def _clean_list(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for s in xs or []:
        if s is None: 
            continue
        s2 = str(s).strip()
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        out.append(s2[:400])  # coupe un peu pour éviter un prompt énorme
    return out


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

stop_fr = set(stopwords.words('french'))

def _tokens(s: str) -> List[str]:
    toks = [t.lower() for t in word_tokenize(s)]
    return [t for t in toks if len(t) >= 3 and t not in stop_fr and t.isalpha()]


