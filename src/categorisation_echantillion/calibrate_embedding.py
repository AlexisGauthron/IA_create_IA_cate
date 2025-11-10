import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)



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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import src.categorisation_echantillion.f_embedding as f_emb


class embedding:
    def __init__(
        self,
        alpha: float | None = None,     # None => alpha adaptatif par classe
        alpha_base: float = 0.30,
        alpha_max_extra: float = 0.40,
        alpha_lam: int = 6,
    ):
        self.alpha = alpha
        self.alpha_base = alpha_base
        self.alpha_max_extra = alpha_max_extra
        self.alpha_lam = alpha_lam

    # --- alpha(n) commun partout ---
    def _alpha_for_count(self, n: int) -> float:
        if self.alpha is not None:
            return float(np.clip(self.alpha, 0.0, 0.99))
        extra = self.alpha_max_extra * (self.alpha_lam / (n + self.alpha_lam))
        return float(np.clip(self.alpha_base + extra, 0.0, 0.85))

    def build_prototypes(
        self,
        shots: Dict[str, List[str]],
        label_defs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Prototype = moyenne des embeddings d'exemples (+ mélange avec définition pondérée par alpha par classe)."""
        protos = {}
        for lbl, examples in shots.items():
            if not examples:
                continue
            ex_vecs = f_emb.embed_texts(examples, is_query=False)
            proto = ex_vecs.mean(axis=0)
            if label_defs and label_defs.get(lbl):
                a = self._alpha_for_count(len(examples))    # <-- plus jamais 1 - None
                d_vec = f_emb._embed_one_cached(label_defs[lbl], is_query=False)
                proto = (1.0 - a) * proto + a * d_vec
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            protos[lbl] = proto.astype(np.float32)
        return protos

    def calibrate_threshold(
        self,
        shots: Dict[str, List[str]],
        label_defs: Optional[Dict[str, str]] = None,
        *,
        perc: int = 10,
        class_balanced: bool = True,
        thr_bounds: Tuple[float, float] = (0.20, 0.60),
        mar_bounds: Tuple[float, float] = (0.02, 0.15),
    ) -> Tuple[float, float]:
        """
        Leave-one-out. On collecte, pour les positifs (bonne classe en top-1):
          - top_sim (similarité top-1),
          - margin = top_sim - second_sim.
        Seuils = percentile `perc` (robuste), équilibrés par classe si demandé, puis clippés.
        L'alpha est adaptatif par classe via self._alpha_for_count.
        """

        def _build_prototypes_alpha(shots_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
            protos = {}
            for lbl, examples in shots_dict.items():
                if not examples:
                    continue
                ex_vecs = f_emb.embed_texts(examples, is_query=False)
                proto = ex_vecs.mean(axis=0)
                if label_defs and label_defs.get(lbl):
                    a = self._alpha_for_count(len(examples))
                    d_vec = f_emb._embed_one_cached(label_defs[lbl], is_query=False)
                    proto = (1.0 - a) * proto + a * d_vec
                proto = proto / (np.linalg.norm(proto) + 1e-8)
                protos[lbl] = proto.astype(np.float32)
            return protos

        per_label_pos_sims: Dict[str, List[float]] = {k: [] for k in shots}
        per_label_margins: Dict[str, List[float]] = {k: [] for k in shots}

        for lbl, examples in shots.items():
            if len(examples) < 2:
                continue
            for i, ex in enumerate(examples):
                others = [t for j, t in enumerate(examples) if j != i]
                tmp_shots = {k: (v if k != lbl else others) for k, v in shots.items()}
                protos = _build_prototypes_alpha(tmp_shots)
                if not protos:
                    continue

                vq = f_emb._embed_one_cached(ex, is_query=True)
                labels = list(protos.keys())
                mats = np.stack([protos[l] for l in labels])
                sims = mats @ vq
                order = np.argsort(-sims)

                if labels[order[0]] == lbl:
                    top_sim = float(sims[order[0]])
                    second_sim = float(sims[order[1]]) if len(order) > 1 else top_sim
                    per_label_pos_sims[lbl].append(top_sim)
                    per_label_margins[lbl].append(top_sim - second_sim)

        def _safe_percentile(xs: List[float], p: int, default: float) -> float:
            return float(np.percentile(xs, p)) if xs else default

        if class_balanced:
            thr_list = [_safe_percentile(per_label_pos_sims[lbl], perc, 0.35) for lbl in shots]
            mar_list = [_safe_percentile(per_label_margins[lbl], perc, 0.05) for lbl in shots]
            thr = float(np.median(thr_list)) if thr_list else 0.35
            mar = float(np.median(mar_list)) if mar_list else 0.05
        else:
            all_pos = [x for xs in per_label_pos_sims.values() for x in xs]
            all_mar = [x for xs in per_label_margins.values() for x in xs]
            thr = _safe_percentile(all_pos, perc, 0.35)
            mar = _safe_percentile(all_mar, perc, 0.05)

        thr = float(np.clip(thr, *thr_bounds))
        mar = float(np.clip(mar, *mar_bounds))
        return thr, mar
