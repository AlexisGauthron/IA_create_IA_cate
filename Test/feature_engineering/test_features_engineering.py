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


from src.features_engineering.helper.lire_json import load_json
from src.analyse.helper.compress_data import compact_llm_snapshot_payload

Nom_Projet = ["cate_metier","avis_client","Titanic_Kaggle","Verbatims"]
Label_Projet = ["label","label","Survived","Categorie"]

all_analyse = []

import src.Data.load_datasets as an

for (nom,label) in zip(Nom_Projet,Label_Projet):

    print("[INFO] Chargement Dataset\n")
    # Chargement dataset
    try:
        df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{nom}")
    except:
        df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{nom}", sep=";")

    analyse = load_json(f"Test/analyse/json/all/test_analyse_metier_report_{nom}.json")

    from src.features_engineering.fonctions_fe.generate_candidate import generate_feature_space
    candidates = generate_feature_space(analyse)

    for c in candidates:
        print(c)

    from src.features_engineering.fonctions_fe.execute_candidate import apply_feature_engineering
    transformed_df = apply_feature_engineering(df_train, analyse)

    print("Tableau",transformed_df)

