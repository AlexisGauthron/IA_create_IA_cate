"""
Pipeline simplifié pour lancer AutoML sur un dataset.
"""

import os
import sys

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path

from src.automl.runner import AutoMLRunner
from src.core.io_utils import csv_to_dataframe_train_test
from src.core.preprocessing import df_to_list_kaggle

# Liste des frameworks AutoML disponibles
AUTOML_FRAMEWORKS = ["flaml", "autogluon", "tpot", "h2o"]


def pipeline_create_model(
    project_name: str,  # Renommé de 'Nom_Projet' → snake_case anglais
    target_col: str,
    automl_frameworks=None,  # Renommé de 'autoML' → snake_case
    data_dir: str = "data/raw",
):
    """
    Pipeline pour créer un modèle via AutoML.

    Args:
        project_name: Nom du projet (anciennement 'Nom_Projet')
        target_col: Colonne cible
        automl_frameworks: Liste des frameworks à utiliser
        data_dir: Dossier des données
    """
    if automl_frameworks is None:
        automl_frameworks = AUTOML_FRAMEWORKS

    print("[INFO] Chargement Dataset\n")

    # Chargement dataset
    data_path = f"{data_dir}/{project_name}"
    try:
        df_train, df_test = csv_to_dataframe_train_test(data_path)
    except Exception:
        df_train, df_test = csv_to_dataframe_train_test(data_path, sep=";")

    # Création du dossier de sortie
    output_dir = f"Modeles/{project_name}"  # Renommé de 'Nom_dossier'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("[INFO] Mise en forme Entrainement\n")

    # Mise en format entrainement
    X_train, X_test, y_train = df_to_list_kaggle(df_train, df_test, target_col)

    # Lancement AutoML
    automl_runner = AutoMLRunner(output_dir, X_train, X_test, y_train)
    automl_runner.use_all(automl_frameworks)
