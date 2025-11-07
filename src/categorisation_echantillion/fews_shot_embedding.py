# src/categorisation_echantillion/embeddings_proto.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
_USE_E5_PREFIX = False
_MODEL_NAME = None


def load_model(name: str = "intfloat/multilingual-e5-base"):
    """Charge et mémorise le modèle d'embeddings."""
    global _model, _USE_E5_PREFIX, _MODEL_NAME
    if _MODEL_NAME == name and _model is not None:
        return _model
    _MODEL_NAME = name
    _USE_E5_PREFIX = name.startswith(("intfloat/", "e5", "gte"))
    _model = SentenceTransformer(name)
    _embed_one_cached.cache_clear()  # on vide le cache si on change de modèle
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


def build_prototypes(
    shots: Dict[str, List[str]],
    label_defs: Optional[Dict[str, str]] = None,
    alpha: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Prototype = moyenne des embeddings d'exemples (+ mélange avec définition pondérée par alpha)."""
    protos = {}
    for lbl, examples in shots.items():
        if not examples:
            continue
        ex_vecs = embed_texts(examples, is_query=False)
        proto = ex_vecs.mean(axis=0)
        if label_defs and label_defs.get(lbl):
            d_vec = _embed_one_cached(label_defs[lbl], is_query=False)
            proto = (1 - alpha) * proto + alpha * d_vec
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        protos[lbl] = proto.astype(np.float32)
    return protos


def classify_one(
    text: str,
    protos: Dict[str, np.ndarray],
    threshold: float = 0.35,
    margin: float = 0.05,
    allow_other: bool = True,
) -> Dict:
    """Mono-label : top-1 par cosinus, rejet 'Autre' si top<seuil ou marge top1-top2 faible."""
    v = _embed_one_cached(text, is_query=True)
    if not protos:
        return {"label": "Autre", "confidence": 0.0, "sims": {}}
    labels = list(protos.keys())
    mats = np.stack([protos[l] for l in labels])
    sims = (mats @ v)
    order = np.argsort(-sims)
    top, second = order[0], (order[1] if len(order) > 1 else order[0])
    top_lbl, top_sim = labels[top], float(sims[top])
    second_sim = float(sims[second])
    if allow_other and (top_sim < threshold or (top_sim - second_sim) < margin):
        return {"label": "Autre", "confidence": max(0.0, min(1.0, top_sim)), "sims": dict(zip(labels, sims.tolist()))}
    return {"label": top_lbl, "confidence": max(0.0, min(1.0, top_sim)), "sims": dict(zip(labels, sims.tolist()))}


def classify_one_multi(
    text: str,
    protos: Dict[str, np.ndarray],
    per_label_threshold: float = 0.4
) -> Dict:
    """Multi-label : conserve toutes les classes dont la similarité dépasse un seuil."""
    v = _embed_one_cached(text, is_query=True)
    sims = {lbl: float(vec @ v) for lbl, vec in protos.items()}
    kept = [lbl for lbl, s in sims.items() if s >= per_label_threshold]
    kept.sort(key=lambda k: sims[k], reverse=True)
    return {"labels": kept if kept else ["Autre"], "sims": sims}


def calibrate_threshold(
    shots: Dict[str, List[str]],
    label_defs: Optional[Dict[str, str]] = None,
    alpha: float = 0.3
) -> Tuple[float, float]:
    """Calibre (threshold, margin) par leave-one-out sur vos exemples."""
    pos_sims, margins = [], []
    for lbl, examples in shots.items():
        if len(examples) < 2:
            continue
        for i, ex in enumerate(examples):
            others = [t for j, t in enumerate(examples) if j != i]
            tmp_shots = {k: (v if k != lbl else others) for k, v in shots.items()}
            protos = build_prototypes(tmp_shots, label_defs, alpha)
            # collecte des stats si bien reconnu
            vq = _embed_one_cached(ex, is_query=True)
            if not protos:
                continue
            labels = list(protos.keys())
            mats = np.stack([protos[l] for l in labels])
            sims = mats @ vq
            order = np.argsort(-sims)
            if labels[order[0]] == lbl:
                top_sim = float(sims[order[0]])
                second_sim = float(sims[order[1]]) if len(order) > 1 else float(sims[order[0]])
                pos_sims.append(top_sim)
                margins.append(top_sim - second_sim)
    thr = float(np.percentile(pos_sims, 10)) if pos_sims else 0.35
    mar = float(np.percentile(margins, 10)) if margins else 0.05
    thr = float(np.clip(thr, 0.2, 0.6))
    mar = float(np.clip(mar, 0.02, 0.15))
    return thr, mar
