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


from collections.abc import Iterable

import numpy as np

import src.few_shot.prototypical.few_shot as fews_prot
import src.few_shot.prototypical.get_definition as get_def


class FewShotManager:
    """
    Orchestrateur :
      - gère plusieurs modèles d'embeddings
      - centralise les grilles d'hyperparamètres
      - gère les définitions de labels
      - lance les sweep() sur chaque modèle
      - permet de récupérer la meilleure config globale
    """

    def __init__(
        self,
        shots: dict[str, list[str]],
        tests: Iterable | None = None,
        model_names: Iterable[str] = (
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
        ),
        *,
        # définitions des labels
        label_defs: dict[str, str] | None = None,
        defs_kwargs: dict | None = None,
        # grilles d'hyperparamètres
        grid_alpha: dict[str, Iterable] | None = None,
        grid_calib: dict[str, Iterable] | None = None,
    ) -> None:
        self.shots = shots
        self.tests = list(tests) if tests is not None else []
        self.model_names = list(model_names)

        # --- defs : on les calcule une seule fois ici ---
        self.defs_kwargs = defs_kwargs or {}

        if label_defs is not None:
            self.label_defs = label_defs
            print(f"[Manager/Defs] {len(self.label_defs)} définitions fournies manuellement.")
        else:
            if not self.shots:
                print("[Manager/Defs] Aucune définition et aucun shots → pas de defs.")
                self.label_defs = None
            else:
                print("[Manager/Defs] Génération automatique des définitions à partir des shots…")
                if hasattr(get_def, "definition_labels_completes"):
                    self.label_defs = get_def.definition_labels_completes(
                        self.shots, **self.defs_kwargs
                    )
                else:
                    # fallback très simple
                    self.label_defs = {
                        lbl: f"Classe '{lbl}' (définition auto basique - manager)"
                        for lbl in self.shots.keys()
                    }
        print("[INFO] Définitions des labels :")
        for lbl, d in self.label_defs.items():
            print(f"  - {lbl} : {d}")

        # --- grilles d'hyperparamètres : une seule source de vérité ---
        # alpha_* : mélange shots / defs
        self.grid_alpha = grid_alpha or {
            "alpha_def": [None],  # None => alpha adaptatif / classe
            "alpha_base": [0.30],
            "alpha_max_extra": [0.40],
            "alpha_lam": [6],
        }

        # calib : perc / balance / bornes
        self.grid_calib = grid_calib or {
            "perc": [10],
            "class_balanced": [True],
            "thr_bounds": [(0.20, 0.60)],
            "mar_bounds": [(0.02, 0.15)],
        }

        # stockage des résultats
        self.results: list[dict] = []
        self.engines: dict[str, fews_prot.PrototypicalFewShot] = {}  # un moteur par modèle

    # ------------------------------------------------------------------ #
    # Lancer les expériences sur tous les modèles
    # ------------------------------------------------------------------ #
    def run_all(
        self,
        *,
        mono_label: bool = True,
        multi_label: bool = False,
        multi_label_floor: float = 0.40,
    ) -> list[dict]:
        """
        Lance un sweep sur tous les modèles d'embeddings.
        Retourne la liste globale de toutes les configs testées (tous modèles).
        """
        self.results.clear()
        self.engines.clear()

        for model_name in self.model_names:
            print("\n" + "#" * 100)
            print(f"[Manager] Modèle embedding = {model_name}")

            # on crée un moteur PrototypicalFewShot pour CE modèle
            engine = fews_prot.PrototypicalFewShot(
                shots=self.shots,
                tests=self.tests,
                model_name=model_name,
                label_defs=self.label_defs,  # les defs sont déjà prêtes
                defs_kwargs={},  # pas besoin de régénérer dans le moteur
            )

            # sweep sur ce modèle uniquement
            rows = engine.sweep(
                grid_alpha=self.grid_alpha,
                grid_calib=self.grid_calib,
                defs_kwargs={},  # defs déjà fixées
                mono_label=mono_label,
                multi_label=multi_label,
                multi_label_floor=multi_label_floor,
            )

            # on garde une référence au moteur pour ce modèle
            self.engines[model_name] = engine

            # on ajoute les résultats au tableau global
            for r in rows:
                # par sécurité, on force la présence du nom du modèle dans la ligne
                r["model_name"] = model_name
                self.results.append(r)

        print(f"\n[Manager] Total de configurations testées (tous modèles) : {len(self.results)}")
        return self.results

    # ------------------------------------------------------------------ #
    # Récupérer la meilleure config globale
    # ------------------------------------------------------------------ #
    def get_best_global(
        self,
        metric: str = "acc_mono",
        *,
        minimize: bool = False,
    ) -> tuple[dict[str, np.ndarray], float | None, float | None, dict]:
        """
        Sélectionne la meilleure configuration (tous modèles confondus)
        selon la métrique choisie, puis :

          - reconstruit les prototypes pour ce modèle + ces hyperparams
          - recalibre (thr, mar)
          - renvoie (protos, thr, mar, config_dict)

        metric : par ex. "acc_mono", "thr", "mar"…
        minimize : False si on maximise (acc), True si on minimise (logloss, etc.)
        """
        if not self.results:
            raise RuntimeError("Aucun résultat en mémoire. Appelle d'abord run_all().")

        # on filtre les lignes qui ont la métrique
        cand = [r for r in self.results if metric in r]
        if not cand:
            raise ValueError(f"Aucune configuration ne contient la métrique '{metric}'.")

        if minimize:
            best_row = min(cand, key=lambda r: r[metric])
        else:
            best_row = max(cand, key=lambda r: r[metric])

        model_name = best_row["model_name"]
        print("\n[Manager] Meilleure config globale :")
        print(f"  - metric={metric} -> value={best_row[metric]}")
        print(f"  - model_name={model_name}")

        # on récupère le moteur correspondant
        engine = self.engines.get(model_name)
        if engine is None:
            raise RuntimeError(f"Aucun moteur enregistré pour model_name={model_name}")

        # on reconstruit via get_prototypes(which='params')
        # avec les hyperparams stockés dans la ligne
        params_alpha = {
            k: best_row.get(k)
            for k in ["alpha_def", "alpha_base", "alpha_max_extra", "alpha_lam"]
            if k in best_row
        }
        params_calib = {
            k: best_row.get(k)
            for k in ["perc", "class_balanced", "thr_bounds", "mar_bounds"]
            if k in best_row
        }
        params = {**params_alpha, **params_calib}

        protos, thr, mar, cfg = engine.get_prototypes(
            which="params",
            params=params,
        )

        # on enrichit un peu la config renvoyée
        cfg = dict(cfg)  # copie
        cfg["model_name"] = model_name
        cfg["metric"] = metric
        cfg["metric_value"] = best_row[metric]

        return protos, thr, mar, cfg

    # ------------------------------------------------------------------ #
    # Affichage résumé global
    # ------------------------------------------------------------------ #
    def print_global_results(
        self,
        sort_by: str = "acc_mono",
        descending: bool = True,
        max_rows: int | None = 50,
    ) -> None:
        """
        Affiche un tableau global de toutes les configs de tous les modèles.
        """
        if not self.results:
            print("[Manager] Aucun résultat. Appelle d'abord run_all().")
            return

        rows = sorted(
            self.results,
            key=lambda r: r.get(sort_by, float("-inf")),
            reverse=descending,
        )

        if max_rows is not None:
            rows = rows[:max_rows]

        col_order = [
            "model_name",
            "defs",
            "alpha_def",
            "alpha_base",
            "alpha_max_extra",
            "alpha_lam",
            "perc",
            "class_balanced",
            "thr_bounds",
            "mar_bounds",
            "thr",
            "mar",
            "acc_mono",
            "ml_thr_used",
        ]

        print("\n=== Résultats globaux (tous modèles) ===\n")
        for i, r in enumerate(rows):
            parts = [f"{k}={r[k]}" for k in col_order if k in r]
            line = " | ".join(parts)
            print(f"[{i:03d}] {line}")
