from __future__ import annotations

import os
import sys

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import multiprocessing

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


# Importation pour mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

import src.few_shot.prototypical.embed_texte as embed_texte


def classify_one(
    text: str,
    protos: dict[str, np.ndarray],
    threshold: float,
    margin: float,
    allow_other: bool,
    embedder: embed_texte.TextEmbedder,
    model_name: str,
) -> dict:
    """Mono-label : top-1 par cosinus, rejet 'Autre' si top<seuil ou marge top1-top2 faible."""
    v = embedder._embed_one_cached(model_name, text, is_query=True)
    if not protos:
        return {"label": "Autre", "confidence": 0.0, "sims": {}}
    labels = list(protos.keys())
    mats = np.stack([protos[l] for l in labels])
    sims = mats @ v
    order = np.argsort(-sims)
    top, second = order[0], (order[1] if len(order) > 1 else order[0])
    top_lbl, top_sim = labels[top], float(sims[top])
    second_sim = float(sims[second])
    if allow_other and (top_sim < threshold or (top_sim - second_sim) < margin):
        return {
            "label": "Autre",
            "confidence": max(0.0, min(1.0, top_sim)),
            "sims": dict(zip(labels, sims.tolist(), strict=False)),
        }
    return {
        "label": top_lbl,
        "confidence": max(0.0, min(1.0, top_sim)),
        "sims": dict(zip(labels, sims.tolist(), strict=False)),
    }


def classify_one_multi(
    text: str,
    protos: dict[str, np.ndarray],
    per_label_threshold: float,
    embedder: embed_texte.TextEmbedder,
    model_name: str,
) -> dict:
    """Multi-label : conserve toutes les classes dont la similarité dépasse un seuil."""
    v = embedder._embed_one_cached(model_name, text, is_query=True)
    sims = {lbl: float(vec @ v) for lbl, vec in protos.items()}
    kept = [lbl for lbl, s in sims.items() if s >= per_label_threshold]
    kept.sort(key=lambda k: sims[k], reverse=True)
    return {"labels": kept if kept else ["Autre"], "sims": sims}
