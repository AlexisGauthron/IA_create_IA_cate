
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



from itertools import product
from typing import Dict, List, Tuple, Optional, Iterable
import textwrap
import numpy as np  

import src.few_shot.embedding.embed_texte as f_emb
import src.few_shot.embedding.calibrate_proto as cal_emb
import src.few_shot.embedding.classify as fews_emb
import src.few_shot.embedding.get_definition as get_def




def _freeze_shots(shots: Dict[str, List[str]]) -> Tuple:
    return tuple((lbl, tuple(shots[lbl])) for lbl in sorted(shots))

def _freeze_defs(defs: Optional[Dict[str, str]]) -> Optional[Tuple]:
    return None if defs is None else tuple(sorted(defs.items()))

class FewShotExperiment:


    def __init__(self,
                 shots: Optional[Dict[str, List[str]]] = None,
                 tests: Optional[Iterable] = None,
                 model_name: str = "intfloat/multilingual-e5-base"):
        
        if shots is None:
            print("[Init] Aucun shots fourni pour l’instant — vous pourrez les ajouter via set_shots()/add_shot().")
            self.shots: Dict[str, List[str]] = {}
        else:
            self.shots = shots
        self.tests = list(tests) if tests is not None else []
        self.model_name = model_name
        self.embedder = f_emb.Embed_textes(self.model_name)

        # état courant
        self.label_defs: Optional[Dict[str, str]] = None
        if shots is not None:
            self._shots_key = _freeze_shots(shots)
        else:
            self._shots_key: Optional[Tuple] = None
        self._defs_key: Optional[Tuple] = None

        # objets/caches
        self._calibrator: Optional[cal_emb.Calibrate_proto] = None
        self._protos = None
        self._protos_key: Optional[Tuple] = None       # clé qui décrit comment les protos ont été construits
        self._thrmar_cache: Dict[Tuple, Tuple[float, float]] = {}

        self.rows = None  # pour le sweep

    def set_shots(self, shots: Dict[str, List[str]]) -> None:
        self.shots = shots
        self._shots_key = _freeze_shots(shots)

    # ---------- Définitions (avec prints) ----------
    def set_definitions(self,
                        allow_defs: bool = False,
                        label_defs: Optional[Dict[str, str]] = None,
                        *,
                        model_defs: str = "mistral:latest",
                        max_terms_defs: int = 10,
                        max_pos_examples_defs: int = 12,
                        max_neg_examples_defs: int = 10,
                        temperature_defs: float = 0.0,
                        seed_defs: Optional[int] = 42) -> None:
        if allow_defs is False and label_defs:
            print("\nUtilisation des définitions de labels fournies…\n")
            defs = label_defs
        elif allow_defs is True:
            print("\nCréation des définitions de labels via LLM…\n")
            defs = get_def.definition_labels_concises(
                self.shots,
                model=model_defs,
                max_terms=max_terms_defs,
                max_pos_examples=max_pos_examples_defs,
                max_neg_examples=max_neg_examples_defs,
                temperature=temperature_defs,
                seed=seed_defs
            )
        else:
            print("\nPas de définitions de labels utilisées.\n")
            defs = None

        self.label_defs = defs
        if defs:
            print("Définitions de labels utilisées:")
            for lbl, defi in defs.items():
                print(f"- {lbl}: {defi}")
            print("")

        # changer les defs invalide les protos / calibrations
        self._defs_key = _freeze_defs(defs)
        self._calibrator = None
        self._protos = None
        self._protos_key = None
        self._thrmar_cache.clear()
        return defs




    # ---------- Prototypes (avec tes prints) ----------
    def build_prototypes(self,
                         *,
                         alpha_def: float | None = None,
                         alpha_base: float = 0.30,
                         alpha_max_extra: float = 0.40,
                         alpha_lam: int = 6):
        key = ("protos",
               self._shots_key,
               self._defs_key,
               self.model_name,
               alpha_def, alpha_base, alpha_max_extra, alpha_lam)

        if key != self._protos_key:
            print("[Build] Construction des prototypes…")
            self._calibrator = cal_emb.Calibrate_proto(
                shots = self.shots,
                label_defs = self.label_defs,
                alpha = alpha_def,                   # None => alpha adaptatif par classe
                alpha_base = alpha_base,
                alpha_max_extra = alpha_max_extra,
                alpha_lam = alpha_lam,
                model_name = self.model_name,
                embedder = self.embedder
            )
            self._protos = self._calibrator.build_prototypes()
            self._protos_key = key
            self._thrmar_cache.clear()           # nouveaux protos => recalibration nécessaire
        else:
            # pas nécessaire mais utile pour suivre
            print("[Build] Prototypes déjà à jour (cache).")

        return self._protos




    # ---------- Calibration (avec tes prints) ----------
    def calibrate(self,
                  *,
                  perc: int = 10,
                  class_balanced: bool = True,
                  thr_bounds: Tuple[float, float] = (0.20, 0.60),
                  mar_bounds: Tuple[float, float] = (0.02, 0.15)) -> Tuple[float, float]:
        if self._calibrator is None:
            raise RuntimeError("Appelle d'abord build_prototypes(...)")

        ckey = ("calib", self._protos_key, perc, class_balanced, thr_bounds, mar_bounds)
        if ckey in self._thrmar_cache:
            thr, mar = self._thrmar_cache[ckey]
        else:
            print("[Calib] Calibration des hyperparamètres de rejet (threshold, margin)…")
            thr, mar = self._calibrator.calibrate_threshold(
                perc=perc,
                class_balanced=class_balanced,
                thr_bounds=thr_bounds,
                mar_bounds=mar_bounds,
            )
            self._thrmar_cache[ckey] = (thr, mar)
        print(f"[Calib] threshold={thr:.3f} | margin={mar:.3f}")
        return thr, mar



    # ---------- Tests (avec tes prints) ----------
    def test_mono(self, threshold: float, margin: float, allow_other: bool = True,
                  tests: Optional[Iterable] = None) -> float:
        tests = list(tests) if tests is not None else self.tests
        print("\n===== Test mono-label (classify_one) =====")
        ok = 0
        for text, expected in tests:
            res = fews_emb.classify_one(text, self._protos, threshold=threshold, margin=margin, allow_other=allow_other,embedder = self.embedder,model_name = self.model_name)
            pred, conf = res["label"], res["confidence"]
            mark = "OK" if pred == expected else "!!"
            top3 = sorted(res["sims"].items(), key=lambda kv: kv[1], reverse=True)[:3]
            sims_str = ", ".join([f"{k}:{v:.2f}" for k, v in top3])
            print(f"[{mark}] y={expected:<18} → ŷ={pred:<18} (conf={conf:.2f}) | top3= {sims_str}")
            ok += int(pred == expected)
        acc = ok / max(1, len(tests))
        print(f"[Score] Exact-match accuracy: {ok}/{len(tests)} = {acc:.1%}")
        return acc


    def test_multi(self, per_label_threshold: float, texts: Optional[Iterable] = None) -> None:
        texts = list(texts) if texts is not None else [t for t, _ in self.tests]  # si tests=(text, y)
        ml_thr = per_label_threshold
        print("\n===== Test multi-label (classify_one_multi) =====")
        for text in texts:
            res = fews_emb.classify_one_multi(text, self._protos, per_label_threshold=ml_thr,embedder = self.embedder,model_name = self.model_name)
            order = sorted(self._protos.keys(), key=lambda k: res["sims"][k], reverse=True)
            top3 = [(k, res["sims"][k]) for k in order[:3]]
            print("- " + textwrap.fill(text, width=88))
            print(f"  → labels={res['labels']} | top3=" + ", ".join(f"{k}:{v:.2f}" for k, v in top3))




    # ---------- Orchestration « une config » (tout-en-un, avec prints) ----------
    def run_once(self,
                 *,
                 mono_label: bool = True,
                 multi_label: bool = False,
                 allow_defs: bool = False,
                 label_defs: Optional[Dict[str, str]] = None,
                 model_defs: str = "mistral:latest",
                 max_terms_defs: int = 10,
                 max_pos_examples_defs: int = 12,
                 max_neg_examples_defs: int = 10,
                 temperature_defs: float = 0.0,
                 seed_defs: Optional[int] = 42,
                 alpha_def: float | None = None,
                 alpha_base: float = 0.30,
                 alpha_max_extra: float = 0.40,
                 alpha_lam: int = 6,
                 perc: int = 10,
                 class_balanced: bool = True,
                 thr_bounds: Tuple[float, float] = (0.20, 0.60),
                 mar_bounds: Tuple[float, float] = (0.02, 0.15),
                 multi_label_floor: float = 0.40) -> Tuple[float, float, Optional[float]]:
        # defs
        self.set_definitions(
            allow_defs=allow_defs, label_defs=label_defs,
            model_defs=model_defs, max_terms_defs=max_terms_defs,
            max_pos_examples_defs=max_pos_examples_defs, max_neg_examples_defs=max_neg_examples_defs,
            temperature_defs=temperature_defs, seed_defs=seed_defs
        )
        # protos
        self.build_prototypes(
            alpha_def=alpha_def, alpha_base=alpha_base,
            alpha_max_extra=alpha_max_extra, alpha_lam=alpha_lam
        )
        # calib
        thr, mar = self.calibrate(
            perc=perc, class_balanced=class_balanced,
            thr_bounds=thr_bounds, mar_bounds=mar_bounds
        )
        # tests
        acc = None
        if mono_label and self.tests:
            acc = self.test_mono(threshold=thr, margin=mar)
        if multi_label:
            ml_thr = max(multi_label_floor, thr)
            self.test_multi(per_label_threshold=ml_thr)
        return thr, mar, acc




    # ---------- Sweep (multi-configs) ----------
    def sweep(self,
              grid_alpha: Dict[str, Iterable],
              grid_calib: Dict[str, Iterable],
              defs_kwargs: Optional[Dict] = None,
              *,
              mono_label: bool = True,
              multi_label: bool = False,
              multi_label_floor: float = 0.40) -> List[Dict]:
        defs_kwargs = defs_kwargs or {}
        rows = []
        # Les defs changent rarement : on les fixe avant
        self.set_definitions(**defs_kwargs)

        for avals in product(*grid_alpha.values()):
            aparams = dict(zip(grid_alpha.keys(), avals))
            print("\n" + "="*86)
            print(f"[Config/Alpha] {aparams}")
            self.build_prototypes(**aparams)

            for cvals in product(*grid_calib.values()):
                cparams = dict(zip(grid_calib.keys(), cvals))
                print(f"[Config/Calib] {cparams}")
                thr, mar = self.calibrate(**cparams)

                row = {"thr": round(thr, 4), "mar": round(mar, 4),
                       **aparams, **cparams}

                if mono_label and self.tests:
                    acc = self.test_mono(threshold=thr, margin=mar)
                    row["acc_mono"] = round(float(acc), 4)

                if multi_label:
                    ml_thr = max(multi_label_floor, thr)
                    self.test_multi(per_label_threshold=ml_thr)
                    row["ml_thr_used"] = round(float(ml_thr), 4)

                rows.append(row)
        return rows

    
    def get_prototypes(self,
                *,
                which: str = "best",            # "best" | "last" | "index" | "params"
                index: Optional[int] = None,
                params: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Optional[float], Optional[float], Dict]:
        """
        Récupère les prototypes + (thr, mar) + la config choisie depuis self.rows,
        reconstruit si nécessaire, et affiche un résumé.

        Retourne: (protos, thr, mar, config_dict)
        """

        # --- sécurité : self.rows doit exister ---
        rows = getattr(self, "rows", None)
        if not rows:
            print("[Recover] Aucun résultat en mémoire (self.rows). Lance d'abord sweep(...).")
            if self._protos is not None:
                # renvoie l'état courant à défaut
                any_thr_mar = next(iter(self._thrmar_cache.values())) if self._thrmar_cache else (None, None)
                return self._protos, any_thr_mar[0], any_thr_mar[1], {
                    "source": "current",
                    "model_name": self.model_name
                }
            return {}, None, None, {}

        # --- résumé global (tes prints) ---
        print(f"\n[Sweep] {len(rows)} configurations testées.")
        col_order = [
            "model_name", "defs",
            "alpha_def", "alpha_base", "alpha_max_extra", "alpha_lam",
            "perc", "class_balanced", "thr_bounds", "mar_bounds",
            "thr", "mar", "acc_mono", "ml_thr_used"
        ]

        def _fmt_row(r: Dict) -> str:
            return " | ".join(f"{k}={r[k]}" for k in col_order if k in r)

        # --- sélection de la config ---
        pick = None
        if which == "best":
            cand = [r for r in rows if "acc_mono" in r]
            if cand:
                pick = max(cand, key=lambda r: r["acc_mono"])
            else:
                pick = rows[-1]
        elif which == "last":
            pick = rows[-1]
        elif which == "index":
            if index is None:
                raise ValueError("get_prototypes(which='index') nécessite index=int.")
            if not (0 <= index < len(rows)):
                raise IndexError(f"index hors limites (0..{len(rows)-1}).")
            pick = rows[index]
        elif which == "params":
            if not params:
                raise ValueError("get_prototypes(which='params') nécessite params=dict.")
            # première ligne qui matche toutes les clés/valeurs de params
            for r in rows:
                if all(r.get(k) == v for k, v in params.items()):
                    pick = r; break
            if pick is None:
                raise ValueError(f"Aucune config ne matche params={params}.")
        else:
            raise ValueError("which doit être parmi {'best','last','index','params'}.")

        # --- print de la meilleure/choisie (tes prints) ---
        print(f"[Sweep] Config choisie -> {_fmt_row(pick)}")

        # --- si le modèle a changé, on switch proprement ---
        if "model_name" in pick and pick["model_name"] != self.model_name:
            if hasattr(self, "set_model"):
                print(f"[Recover] Switch model -> {pick['model_name']}")
                self.set_model(pick["model_name"])
            else:
                print(f"[Recover] Switch model (fallback) -> {pick['model_name']}")
                # fallback si pas de set_model
                self.model_name = pick["model_name"]
                self.embedder.load_model(self.model_name)
                self._calibrator = None
                self._protos = None
                self._protos_key = None
                self._thrmar_cache.clear()

        # --- reconstruire protos & recalibrer avec EXACTEMENT ces hyperparams ---
        aparams = {k: pick.get(k) for k in ["alpha_def", "alpha_base", "alpha_max_extra", "alpha_lam"] if k in pick}
        self.build_prototypes(**aparams)

        cparams = {k: pick.get(k) for k in ["perc", "class_balanced", "thr_bounds", "mar_bounds"] if k in pick}
        thr, mar = self.calibrate(**cparams)

        # On renvoie l'état reconstruit + la config choisie
        return self._protos, thr, mar, pick



    def print_all_results(self, sort_by: Optional[str] = "acc_mono", descending: bool = True) -> None:

        if not self.rows:
            print("[Sweep] Aucun résultat en mémoire. Lance d'abord sweep(...).")
            return

        if sort_by is not None:
            rows = sorted(self.rows, key=lambda r: r.get(sort_by, float("-inf")), reverse=descending)

        col_order = [
            "model_name", "defs",
            "alpha_def", "alpha_base", "alpha_max_extra", "alpha_lam",
            "perc", "class_balanced", "thr_bounds", "mar_bounds",
            "thr", "mar", "acc_mono", "ml_thr_used"
        ]
        for i, r in enumerate(rows):
            line = " | ".join(f"{k}={r[k]}" for k in col_order if k in r)
            print(f"[{i:03d}] {line}")