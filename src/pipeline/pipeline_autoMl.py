import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.automl.runner import all_autoML
from src.core.io_utils import csv_to_dataframe_train_test
from src.core.preprocessing import df_to_list_Kaggle


model_autoML = ["flaml","autogluon","tpot","h2o"]

def pipeline_create_model(Nom_Projet: str, target_col, autoML=model_autoML, data_dir: str = "data/raw"):

    print("[INFO] Chargement Dataset\n")
    # Chargement dataset
    data_path = f"{data_dir}/{Nom_Projet}"
    try:
        df_train, df_test = csv_to_dataframe_train_test(data_path)
    except:
        df_train, df_test = csv_to_dataframe_train_test(data_path, sep=";")


    # chargement model
    Nom_dossier = f"Modeles/{Nom_Projet}"
    dossier = Path(Nom_dossier)
    dossier.mkdir(parents=True, exist_ok=True)


    print("[INFO] Mise en forme Entrainement\n")
    # Mise en format entrainement
    X_train, X_test, y_train = df_to_list_Kaggle(df_train, df_test, target_col)

    automl_runner = all_autoML(Nom_dossier, X_train, X_test, y_train)
    automl_runner.use_all(autoML)


