import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


# Importation pour mac 
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from typing import Dict, List, Tuple, Optional
import numpy as np


import src.few_shot.prototypical.embed_texte as embed_texte




class Calibrate_proto:


    def __init__(
        self,
        shots: Dict[str, List[str]],
        *,

        label_defs: Optional[Dict[str, str]] = None,

        alpha: float | None = None,     # None => alpha adaptatif par classe
        alpha_base: float = 0.30,
        alpha_max_extra: float = 0.40,
        alpha_lam: int = 6,

        model_name: str,
        embedder: embed_texte.Embed_textes,
        ):

        self.shots = shots
        self.label_defs = label_defs
        self.alpha = alpha
        self.alpha_base = alpha_base
        self.alpha_max_extra = alpha_max_extra
        self.alpha_lam = alpha_lam
        self.model_name = model_name
        self.embedder = embedder


    # --- alpha(n) commun partout ---
    def _alpha_for_count(self, n: int) -> float:
        if self.alpha is not None:
            return float(np.clip(self.alpha, 0.0, 0.99))
        
        extra = self.alpha_max_extra * (self.alpha_lam / (n + self.alpha_lam))
        return float(np.clip(self.alpha_base + extra, 0.0, 0.85))


    def build_prototypes(self) -> Dict[str, np.ndarray]:
        """Prototype = moyenne des embeddings d'exemples (+ mélange avec définition pondérée par alpha par classe)."""
        protos = {}
        for lbl, examples in self.shots.items():
            if not examples:
                continue
            ex_vecs = self.embedder.embed_texts(examples, is_query=False)
            proto = ex_vecs.mean(axis=0)
            if self.label_defs and self.label_defs.get(lbl):
                a = self._alpha_for_count(len(examples))    # <-- plus jamais 1 - None
                d_vec = self.embedder._embed_one_cached(self.model_name,self.label_defs[lbl], is_query=False)
                proto = (1.0 - a) * proto + a * d_vec
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            protos[lbl] = proto.astype(np.float32)
        return protos



    def calibrate_threshold(
            self,
            perc: int,                         # Percentile utilisé pour fixer les seuils (ex: 10 → 10e percentile = robuste aux outliers)
            class_balanced: bool,            # Si True : calcule un percentile par classe, puis médiane entre classes (équilibre les classes rares)
            thr_bounds: Tuple[float, float],  # Bornes min/max du seuil de similarité
            mar_bounds: Tuple[float, float],  # Bornes min/max du seuil de marge
        ) -> Tuple[float, float]:
        """
        Calibre deux seuils (thr, mar) pour une décision "accepter la prédiction vs. s'abstenir",
        à partir d'un classifieur par prototypes sur embeddings.

        Méthode :
        - Leave-One-Out (LOO) sur les exemples de chaque classe :
            pour chaque exemple, on reconstruit les prototypes SANS cet exemple
            (évite de s'auto-évaluer sur un prototype qui le contient).
        - On ne collecte des stats que pour les cas "positifs" (la vraie classe est top-1).
            * top_sim   = similarité du top-1
            * margin    = top_sim - second_sim (écart avec le 2e)
        - Les seuils finaux sont des percentiles (robustes) de ces distributions,
            équilibrés par classe si demandé, puis "clippés" dans des bornes sûres.

        Utilisation typique après calibration :
            accepter si (top_sim >= thr) ET ((top_sim - second_sim) >= mar), sinon s'abstenir / "Autre".

        Remarques :
        - Les prototypes par classe mélangent (au besoin) la moyenne des exemples et
            une "définition" textuelle de la classe via un alpha dépendant du nombre d'exemples.
        - On suppose que f_emb.* renvoie des embeddings L2-normalisés ; le produit scalaire ≈ cosinus.
        """

        # ------------------------------------------------------------
        # Helper : construit un prototype par classe avec "alpha-mix"
        #          entre la moyenne des embeddings d'exemples et la
        #          définition textuelle de la classe (si fournie).
        #          Le prototype est L2-normalisé.
        # ------------------------------------------------------------
        def _build_prototypes_alpha(shots_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
            protos = {}
            for lbl, examples in shots_dict.items():
                if not examples:
                    continue

                # Embeddings des exemples de la classe
                ex_vecs = self.embedder.embed_texts(examples, is_query=False)
                proto = ex_vecs.mean(axis=0)  # prototype = moyenne

                # S'il existe une définition textuelle de la classe,
                # on l'entremêle avec un poids alpha dépendant du nb d'exemples
                if self.label_defs and self.label_defs.get(lbl):
                    a = self._alpha_for_count(len(examples))           # alpha ∈ [0, 1], ↑ si classe rare
                    d_vec = self.embedder._embed_one_cached(self.model_name,self.label_defs[lbl], is_query=False)
                    proto = (1.0 - a) * proto + a * d_vec

                # Normalisation L2 (nécessaire pour un produit scalaire = cosinus)
                proto = proto / (np.linalg.norm(proto) + 1e-8)
                protos[lbl] = proto.astype(np.float32)
            return protos

        # Stocke, par classe, les métriques "positives" collectées en LOO
        per_label_pos_sims: Dict[str, List[float]] = {k: [] for k in self.shots}   # top_sim lorsque le top-1 est correct
        per_label_margins: Dict[str, List[float]] = {k: [] for k in self.shots}    # margin = top1 - top2 (idem)

        # ------------------------------------------------------------
        # Boucle Leave-One-Out :
        #   pour chaque exemple d'une classe, le retirer, reconstruire
        #   les prototypes, prédire, et si correct → collecter métriques.
        # ------------------------------------------------------------
        for lbl, examples in self.shots.items():
            if len(examples) < 2:
                # Nécessite ≥ 2 exemples pour faire du LOO sur cette classe
                continue

            for i, ex in enumerate(examples):
                # Retirer l'exemple courant de sa classe
                others = [t for j, t in enumerate(examples) if j != i]
                tmp_shots = {k: (v if k != lbl else others) for k, v in self.shots.items()}

                # Prototypes avec alpha-mix + normalisation
                protos = _build_prototypes_alpha(tmp_shots)
                if not protos:
                    continue

                # Embedding de la requête (exemple tenu-out)
                vq = self.embedder._embed_one_cached(self.model_name,ex, is_query=True)

                # Matrice [n_classes, dim] et similarités cosinus
                labels = list(protos.keys())
                mats = np.stack([protos[l] for l in labels])
                sims = mats @ vq                              # produit scalaire = cosinus si L2-norm.

                # Classements décroissants de similarité
                order = np.argsort(-sims)

                # On ne garde les stats que si la vraie classe est bien top-1
                if labels[order[0]] == lbl:
                    top_sim = float(sims[order[0]])
                    second_sim = float(sims[order[1]]) if len(order) > 1 else top_sim
                    per_label_pos_sims[lbl].append(top_sim)
                    per_label_margins[lbl].append(top_sim - second_sim)



        # Petit utilitaire : percentile sécurisé avec valeur par défaut si liste vide
        def _safe_percentile(xs: List[float], p: int, default: float) -> float:
            return float(np.percentile(xs, p)) if xs else default

        # ------------------------------------------------------------
        # Agrégation robuste par percentile
        #   - class_balanced=True : percentile par classe puis médiane,
        #     ainsi les classes fréquentes ne dominent pas.
        #   - False : on pool tout et on prend un seul percentile global.
        # ------------------------------------------------------------
        if class_balanced:
            thr_list = [_safe_percentile(per_label_pos_sims[lbl], perc, 0.35) for lbl in self.shots]
            mar_list = [_safe_percentile(per_label_margins[lbl], perc, 0.05) for lbl in self.shots]
            thr = float(np.median(thr_list)) if thr_list else 0.35
            mar = float(np.median(mar_list)) if mar_list else 0.05
        else:
            all_pos = [x for xs in per_label_pos_sims.values() for x in xs]
            all_mar = [x for xs in per_label_margins.values() for x in xs]
            thr = _safe_percentile(all_pos, perc, 0.35)
            mar = _safe_percentile(all_mar, perc, 0.05)

        # Clip final dans des bornes "raisonnables" pour éviter des seuils extrêmes
        thr = float(np.clip(thr, *thr_bounds))
        mar = float(np.clip(mar, *mar_bounds))
        return thr, mar
