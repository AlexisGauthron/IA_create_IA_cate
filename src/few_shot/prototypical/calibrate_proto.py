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

# =============================================================================
# Classe PrototypeCalibrator
# =============================================================================
# Renommée de 'Calibrate_proto' vers 'PrototypeCalibrator'
# Raison: 'Calibrate_proto' utilisait un underscore dans le nom de classe
#         Les classes doivent être en PascalCase sans underscore
# =============================================================================


class PrototypeCalibrator:
    """Calibrateur pour les prototypes few-shot."""

    def __init__(
        self,
        shots: dict[str, list[str]],
        *,
        label_defs: dict[str, str] | None = None,
        alpha: float | None = None,  # None => alpha adaptatif par classe
        alpha_base: float = 0.30,
        alpha_max_extra: float = 0.40,
        alpha_lam: int = 6,
        model_name: str,
        embedder: embed_texte.TextEmbedder,
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

    def build_prototypes(self) -> dict[str, np.ndarray]:
        """Prototype = moyenne des embeddings d'exemples (+ mélange avec définition pondérée par alpha par classe)."""
        protos = {}
        for lbl, examples in self.shots.items():
            if not examples:
                continue
            ex_vecs = self.embedder.embed_texts(examples, is_query=False)
            proto = ex_vecs.mean(axis=0)
            if self.label_defs and self.label_defs.get(lbl):
                a = self._alpha_for_count(len(examples))  # <-- plus jamais 1 - None
                d_vec = self.embedder._embed_one_cached(
                    self.model_name, self.label_defs[lbl], is_query=False
                )
                proto = (1.0 - a) * proto + a * d_vec
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            protos[lbl] = proto.astype(np.float32)
        return protos

    def calibrate_threshold(
        self,
        perc: int,
        class_balanced: bool,
        thr_bounds: tuple[float, float],
        mar_bounds: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Même logique que ta version actuelle, mais :
        - embeddings des shots calculés une seule fois
        - prototypes recalculés par somme/moyenne (pas de ré-encode).
        """

        # ------------------------------------------------------------
        # 1) Pré-calcul des embeddings de tous les shots
        # ------------------------------------------------------------
        shot_vecs: dict[str, np.ndarray] = {}
        shot_sums: dict[str, np.ndarray] = {}
        shot_counts: dict[str, int] = {}

        # On encode chaque shot UNE fois
        for lbl, examples in self.shots.items():
            if not examples:
                continue
            ex_vecs = self.embedder.embed_texts(examples, is_query=False)  # [n_lbl, d]
            ex_vecs = np.asarray(ex_vecs, dtype=np.float32)
            shot_vecs[lbl] = ex_vecs
            shot_sums[lbl] = ex_vecs.sum(axis=0)
            shot_counts[lbl] = ex_vecs.shape[0]

        # Embeddings des définitions textuelles (si présentes)
        label_def_vecs: dict[str, np.ndarray] = {}
        if self.label_defs:
            for lbl, text in self.label_defs.items():
                if text:
                    # _embed_one_cached → déjà optimisé / mis en cache
                    label_def_vecs[lbl] = self.embedder._embed_one_cached(
                        self.model_name, text, is_query=False
                    )

        # Stats collectées
        per_label_pos_sims: dict[str, list[float]] = {k: [] for k in self.shots}
        per_label_margins: dict[str, list[float]] = {k: [] for k in self.shots}

        # ------------------------------------------------------------
        # 2) Boucle Leave-One-Out :
        #    pour chaque exemple, on retire son vecteur de la moyenne
        #    de sa classe, et on crée les prototypes à partir de sommes.
        # ------------------------------------------------------------
        for lbl, examples in self.shots.items():
            n_lbl = len(examples)
            if n_lbl < 2:
                # pas possible de faire du LOO avec < 2 exemples
                continue
            if lbl not in shot_vecs:
                continue

            vecs_lbl = shot_vecs[lbl]  # [n_lbl, d]
            sum_lbl = shot_sums[lbl]  # [d]

            for i, ex in enumerate(examples):
                # Construire les prototypes de TOUTES les classes
                protos: dict[str, np.ndarray] = {}

                for lbl2, ex_vecs2 in shot_vecs.items():
                    cnt2 = shot_counts[lbl2]

                    # Cas de la classe dont on retire l'exemple i
                    if lbl2 == lbl:
                        if n_lbl <= 1:
                            continue  # sécurité
                        # moyenne des "others" = (somme - vecteur_i) / (n-1)
                        sum_others = sum_lbl - vecs_lbl[i]
                        cnt_others = n_lbl - 1
                        if cnt_others <= 0:
                            continue
                        mean_vec = sum_others / float(cnt_others)
                        effective_count = cnt_others
                    else:
                        # Classes non modifiées : moyenne simple
                        mean_vec = shot_sums[lbl2] / float(cnt2)
                        effective_count = cnt2

                    # alpha-mix avec la définition, si dispo
                    proto = mean_vec
                    d_vec = label_def_vecs.get(lbl2)
                    if d_vec is not None:
                        a = self._alpha_for_count(effective_count)  # dépend du nb d'exemples
                        proto = (1.0 - a) * proto + a * d_vec

                    # Normalisation L2
                    proto = proto / (np.linalg.norm(proto) + 1e-8)
                    protos[lbl2] = proto.astype(np.float32)

                if not protos:
                    continue

                # Embedding de la requête (exemple tenu-out)
                vq = self.embedder._embed_one_cached(self.model_name, ex, is_query=True)

                labels = list(protos.keys())
                mats = np.stack([protos[l] for l in labels])  # [n_classes, d]
                sims = mats @ vq  # cosinus

                order = np.argsort(-sims)
                if labels[order[0]] == lbl:
                    top_sim = float(sims[order[0]])
                    second_sim = float(sims[order[1]]) if len(order) > 1 else top_sim
                    per_label_pos_sims[lbl].append(top_sim)
                    per_label_margins[lbl].append(top_sim - second_sim)

            print(
                f"[Calib_proto-fast] Classe '{lbl}': {len(per_label_pos_sims[lbl])} positifs sur {len(examples)}"
            )

        # ------------------------------------------------------------
        # 3) Percentiles + agrégation comme avant
        # ------------------------------------------------------------
        def _safe_percentile(xs: list[float], p: int, default: float) -> float:
            return float(np.percentile(xs, p)) if xs else default

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

        thr = float(np.clip(thr, *thr_bounds))
        mar = float(np.clip(mar, *mar_bounds))
        return thr, mar
