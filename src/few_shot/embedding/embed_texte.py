# src/categorisation_echantillion/embeddings_proto.py
import os

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


# Importation pour mac 
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from functools import lru_cache
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class Embed_textes:
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base",
                 device: Optional[str] = None, threads: int = 1):
        
        torch.set_num_threads(max(1, int(threads)))
        self.model_name: str = model_name
        self.device: str = device or self._autodetect_device()
        self.model: Optional[SentenceTransformer] = None
        self.use_e5_prefix: bool = False
        self.load_model(model_name)


    def _autodetect_device(self) -> str:
        try:
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


    def load_model(self, name: Optional[str] = None) -> SentenceTransformer:
        name = name or self.model_name
        if self.model is not None and self.model_name == name:
            return self.model

        self.model_name = name
        self.use_e5_prefix = name.startswith(("intfloat/", "e5", "gte"))

        last_err = None
        for _ in range(3):
            try:
                print(f"Chargement modèle '{name}' sur {self.device}…")
                self.model = SentenceTransformer(name, device=self.device)
                # on vide le cache quand le modèle change
                self._embed_one_cached.cache_clear()
                return self.model
            except RuntimeError as e:
                import time; time.sleep(1)
                last_err = e
        raise RuntimeError(f"Échec de chargement du modèle '{name}': {last_err}")

    def _prep_text(self, txt: str, is_query: bool) -> str:
        if self.use_e5_prefix:
            return f"{'query' if is_query else 'passage'}: {txt}"
        return txt


    @lru_cache(maxsize=8192)
    def _embed_one_cached(self, model_key: str, txt: str, is_query: bool) -> np.ndarray:
        # garantit que le cache n’est pas obsolète si le modèle a changé
        if self.model is None or self.model_name != model_key:
            self.load_model(model_key)
        v = self.model.encode([self._prep_text(txt, is_query)], normalize_embeddings=True)[0]
        return v.astype(np.float32)


    def embed_one(self, txt: str, is_query: bool = False) -> np.ndarray:
        return self._embed_one_cached(self.model_name, txt, is_query)


    def embed_texts(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self.model is None:
            self.load_model()
        prepped = [self._prep_text(t, is_query) for t in texts]
        vs = self.model.encode(prepped, normalize_embeddings=True)
        return vs.astype(np.float32)
