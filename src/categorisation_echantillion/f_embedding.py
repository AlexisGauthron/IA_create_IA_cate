# src/categorisation_echantillion/embeddings_proto.py
import os

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
_USE_E5_PREFIX = False
_MODEL_NAME = None

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



def load_model(name: str = "intfloat/multilingual-e5-base"):
    """Charge et mémorise le modèle d'embeddings de manière sécurisée."""

    import torch
    torch.set_num_threads(1)

    global _model, _USE_E5_PREFIX, _MODEL_NAME
    try:
        # Si le modèle demandé est déjà chargé, on le retourne directement
        if _MODEL_NAME == name and _model is not None:
            return _model

        _MODEL_NAME = name
        _USE_E5_PREFIX = name.startswith(("intfloat/", "e5", "gte"))

        
        import time
        # Chargement du modèles
        for _ in range(3):
            try:
                print("Chargement Model ...\n ")
                _model = SentenceTransformer(name, device='cpu')
                print("Réussie\n")
                break
            except RuntimeError as e:
                print("Retrying model load due to RuntimeError:", e)
                time.sleep(1)


        # On vide le cache de _embed_one_cached si on change de modèle
        _embed_one_cached.cache_clear()
        return _model

    except RuntimeError as e:
        print(f"Erreur lors du chargement du modèle (RuntimeError) : {e}")
    except Exception as e:
        print(f"Erreur inattendue lors du chargement du modèle : {e}")

    # Si échec, on retourne None et on évite le crash
    _model = None
    return _model



def _prep_text(txt: str, is_query: bool) -> str:
    if _USE_E5_PREFIX:
        return f"{'query' if is_query else 'passage'}: {txt}"
    return txt


@lru_cache(maxsize=8192)
def _embed_one_cached(txt: str, is_query: bool) -> np.ndarray:
    if _model is None:
        load_model()  # par défaut
    v = _model.encode([_prep_text(txt, is_query)], normalize_embeddings=True)[0]
    return v.astype(np.float32)


def embed_texts(texts: List[str], is_query: bool) -> np.ndarray:
    if _model is None:
        load_model()
    prepped = [_prep_text(t, is_query) for t in texts]
    vs = _model.encode(prepped, normalize_embeddings=True)
    return vs.astype(np.float32)