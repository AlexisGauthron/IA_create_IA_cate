
import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



from typing import Dict, List, Tuple, Optional
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# Pour chaque prototype, on garde les labels dont la similarité dépasse le seuil.
# Permet d’attribuer plusieurs labels à un texte.
# Si aucun label ne passe le seuil → "Autre".


# --------- Jeu d'exemples (few-shots) par classe ----------

import src.Data.load_datasets as an

Nom_Projet = "cate_metier"
print("[INFO] Chargement Dataset\n")
# Chargement dataset
df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{Nom_Projet}")

shots = df_train.groupby("label")["text"].apply(list).to_dict()
print(f"[INFO] Shots par classe : { {k: len(v) for k,v in shots.items()} }")

tests = df_test

if __name__ == "__main__":
    # Petit banc d'essai autonome pour les fonctions du module.

    import src.few_shot.prototypical.few_shot as few_shot_module
    exp = few_shot_module.FewShotExperiment(shots, tests, model_name="intfloat/multilingual-e5-large")

    # 1) Un run unique (garde tous les prints) avec définitions créeres via LLM
    thr, mar, acc = exp.run_once(
        mono_label=True, multi_label=False,
        allow_defs=True, label_defs=None,        # ou allow_defs=True pour génération LLM
        alpha_def=None, alpha_base=0.30, alpha_max_extra=0.40, alpha_lam=6,
        perc=10, class_balanced=True,
        thr_bounds=(0.20, 0.60), mar_bounds=(0.02, 0.15),
    )

    # 2) Un sweep de configs (affiche un header pour chaque combo + tes logs habituels)
    # grid_alpha = {"alpha_def":[None, 0.2], "alpha_base":[0.25, 0.30],
    #             "alpha_max_extra":[0.30, 0.40], "alpha_lam":[4, 6]}
    # grid_calib = {"perc":[5, 10], "class_balanced":[True],
    #             "thr_bounds":[(0.2,0.6)], "mar_bounds":[(0.02,0.15)]}

    # exp.sweep(grid_alpha, grid_calib, defs_kwargs={"allow_defs": False, "label_defs": label_defs})
    # print("\n=== Résumé final ===\n\n")
    # protos, thr, mar, cfg = exp.get_prototypes(which="best")
    
    # print("\n=== Résumé final ===")
    # print(f"Meilleure config : {cfg}")

    # exp.print_all_results()

