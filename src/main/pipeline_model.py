import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path
from sklearn.model_selection import train_test_split

import src.autoML_supervise.all_autoML as auto_all
import src.Data.load_datasets as an
import src.fonctions.format_entrainement as format_ent


def pipeline_create_model(Nom_Projet : str, target_col, model = ["flaml","autogluon","tpot","h2o"]):

    print("[INFO] Chargement Dataset\n")
    # Chargement dataset
    df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{Nom_Projet}")

    # chargement model
    Nom_dossier = f"Modeles/{Nom_Projet}"
    dossier = Path(Nom_dossier)          # remplace par ton chemin
    dossier.mkdir(parents=True, exist_ok=True)

    print("[INFO] Mise en forme Entrainement\n")
    # Mise en format entrainement 
    # X_train, X_test, y_train = format_ent.df_to_list_Kaggle(df_train,df_test,target_col)

    X_train, X_test, y_train,y_test = format_ent.df_to_list(df_train,target_col)
    
    all = auto_all.all_autoML(Nom_dossier,X_train, X_test, y_train, y_test)
    all.use_all(model)


