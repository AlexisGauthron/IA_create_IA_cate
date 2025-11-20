from typing import Union, List
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Charge le modèle d'embedding une seule fois (cache en mémoire).
    Change `model_name` si tu veux utiliser un autre modèle (E5, BGE, etc.).
    """
    return SentenceTransformer(model_name)


def embed_text(
    text: Union[str, List[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> np.ndarray:
    """
    Génère un embedding à partir d'un texte (ou d'une liste de textes).

    Paramètres
    ----------
    text : str ou List[str]
        Le texte à encoder. Si une liste est fournie, renvoie un tableau (N, dim).
    model_name : str
        Nom du modèle sentence-transformers à utiliser.
    normalize : bool
        Si True, normalise les embeddings (norme L2 = 1), utile pour le cosine.

    Retour
    ------
    np.ndarray
        Embedding du texte. Shape = (dim,) pour une string, ou (N, dim) pour une liste.
    """
    model = _load_embedding_model(model_name)
    emb = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return emb
