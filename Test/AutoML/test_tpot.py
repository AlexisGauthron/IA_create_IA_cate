import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path
from sklearn.model_selection import train_test_split

import src.autoML_supervise.tpot1 as auto_tpot
import src.Data.load_datasets as an


if __name__ == "__main__":

    # Chargement_modele
    Nom_Projet = "breast_cancer"

    Nom_dossier = f"Modeles/python/{Nom_Projet}"
    dossier = Path(Nom_dossier)          # remplace par ton chemin
    dossier.mkdir(parents=True, exist_ok=True)


    # Chargement dataset
    df = an.load_datasets_breast_cancer()

    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    Tpot = auto_tpot.autoMl_tpot(Nom_dossier,X_train, X_test, y_train, y_test)
    Tpot.tpot1()
    Tpot.predict_test()

    # Enregistrer le model
    Tpot.enregistrement_model()

