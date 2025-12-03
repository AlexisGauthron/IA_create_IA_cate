# text_transforms.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features_engineering.transformation_fe.registry import register
import re
from collections import Counter


@register("tfidf")
def tfidf(text, n_components=2048):
    """
    TF-IDF robuste : gère les valeurs manquantes et crée un DataFrame.
    
    Paramètres :
    - text : pd.Series de type string (peut contenir NaN)
    - n_components : nombre de features TF-IDF
    """
    if not isinstance(text, pd.Series):
        raise TypeError(f"tfidf requires a pandas Series, got {type(text)}")

    # Remplacer NaN par chaîne vide
    text_clean = text.fillna("")

    vec = TfidfVectorizer(max_features=n_components)
    arr = vec.fit_transform(text_clean).toarray()

    return pd.DataFrame(arr, columns=[f"tfidf_{i}" for i in range(arr.shape[1])])



@register("hashing_ngrams")
def hashing_ngrams(text_col, n_components = 1024):
    """
    Hashing trick basé sur des ngrams (1–3) sur du texte.
    Version simplifiée du HashingVectorizer.
    """
    if not isinstance(text_col, pd.Series):
        raise TypeError(f"hashing_ngrams() requires a pandas Series, got {type(text_col)}")

    n_components = int(n_components)

    def tokenize(txt):
        if not isinstance(txt, str):
            return []
        return re.findall(r"\w+", txt.lower())

    def make_ngrams(tokens):
        ngrams = []
        L = len(tokens)
        # unigram + bigram + trigram
        for i in range(L):
            ngrams.append(tokens[i])
            if i + 1 < L:
                ngrams.append(tokens[i] + "_" + tokens[i+1])
            if i + 2 < L:
                ngrams.append(tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2])
        return ngrams

    # Matrice finale
    rows = []

    for txt in text_col:
        tokens = tokenize(txt)
        ng = make_ngrams(tokens)
        hashed_counts = Counter([hash(t) % n_components for t in ng])

        row = [hashed_counts.get(i, 0) for i in range(n_components)]
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=[f"hash_ngram_{text_col.name}_{i}" for i in range(n_components)]
    )

    return df