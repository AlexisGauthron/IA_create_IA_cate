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
import json 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# Pour chaque prototype, on garde les labels dont la similarité dépasse le seuil.
# Permet d’attribuer plusieurs labels à un texte.
# Si aucun label ne passe le seuil → "Autre".


# --------- Jeu d'exemples (few-shots) par classe ----------

import src.Data.load_datasets as an

# Nom_Projet = ["cate_metier","avis_client","Titanic_Kaggle","Verbatims"]
# Label_Projet = ["label","label","Survived","Categorie"]

Nom_Projet = ["Verbatims"]
Label_Projet = ["Categorie"]

# Nom_Projet = ["Titanic_Kaggle"]
# Label_Projet = ["Survived"]

for (nom,label) in zip(Nom_Projet,Label_Projet):

    print("[INFO] Chargement Dataset\n")
    # Chargement dataset
    try:
        df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{nom}")
    except:
        df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{nom}", sep=";")


    from src.analyse.analyse import analyse

    analyse_propre = analyse(df_train,label,nom,print_json=True,chemin_json = "test",model_metier="mistral:7b-instruct")
    

        

