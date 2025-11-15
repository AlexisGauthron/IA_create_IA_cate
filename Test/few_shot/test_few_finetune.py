
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



from typing import Dict, List, Tuple, Optional
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# Pour chaque prototype, on garde les labels dont la similarité dépasse le seuil.
# Permet d’attribuer plusieurs labels à un texte.
# Si aucun label ne passe le seuil → "Autre".


# --------- Jeu d'exemples (few-shots) par classe ----------

import src.Data.load_datasets as an
import src.few_shot.few_shot_finetune as few_shot_finetune_module

Nom_Projet = "cate_metier"
print("[INFO] Chargement Dataset\n")
# Chargement dataset
df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{Nom_Projet}")

modele_util_finetune = "sentence-transformers/all-MiniLM-L6-v2"

modele_finetune, label2id, id2label = few_shot_finetune_module.few_shot_finetune(
    train_df=df_train,
    dev_df=df_test,
    model_name=modele_util_finetune
)

shots = df_train.groupby("label")["text"].apply(list).to_dict()
print(f"[INFO] Shots par classe : { {k: len(v) for k,v in shots.items()} }")

tests = list(zip(df_test["text"], df_test["label"]))

if __name__ == "__main__":
    # Petit banc d'essai autonome pour les fonctions du module.

    import src.few_shot.few_shot_all as few_shot_all
    
    # 3) Liste de modèles d'embedding à comparer
    model_names = (
        # "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        # "models/e5-multiling-finetune-v1",   # ← ton modèle finetuné sauvegardé en local
    )

    # 4) Paramètres pour la génération automatique des définitions (via get_def)
    defs_kwargs = {
        "model": "mistral:7b-instruct",  # adapté à ton backend (Ollama, API, etc.)
        "max_pos_examples" : 8,
        "max_neg_examples" : 6,
        "max_terms" : 12,
        "temperature" : 0.1,
        "seed" : 42,
        "batch_size" : 5,
    }

    # 5) Grille d'hyperparamètres pour les prototypes
    grid_alpha = {
        "alpha_def": [None],   # None = alpha adaptatif, 0.3 / 0.6 = mélange fixe
        "alpha_base": [0.30],
        "alpha_max_extra": [0.30],
        "alpha_lam": [8],
    }

    # 6) Grille d'hyperparamètres pour la calibration threshold/margin
    grid_calib = {
        "perc": [5],
        "class_balanced": [True],
        "thr_bounds": [(0.2,0.5), (0.20, 0.60), (0.0, 0.70),(0.0, 0.8)],
        "mar_bounds": [(0.00, 0.4)],
    }

    # 7) Instanciation du manager AVEC defs automatiques
    manager = few_shot_all.FewShotManager(
        shots=shots,
        tests=tests,
        model_names=model_names,
        label_defs=None,        # ← on laisse le manager + get_def générer les définitions
        defs_kwargs=defs_kwargs,
        grid_alpha=grid_alpha,
        grid_calib=grid_calib,
    )

    # 1) on lance toutes les expériences (tous modèles, toutes configs)
    manager.run_all(mono_label=True, multi_label=False)

    # 2) on récupère la meilleure config globale
    protos, thr, mar, cfg = manager.get_best_global(metric="acc_mono")

    print("\n[FINAL]")
    print("Meilleur modèle :", cfg["model_name"])
    print("thr, mar        :", thr, mar)
    print("acc_mono        :", cfg.get("acc_mono"))


