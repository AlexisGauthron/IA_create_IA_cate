import sys
import os
import warnings

# Supprime les warnings non critiques de sklearn
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.automl.runner import all_autoML
from src.core.io_utils import load_datasets_breast_cancer


if __name__ == "__main__":

    # Chargement_modele
    Nom_Projet = "breast_cancer"

    Nom_dossier = f"Modeles/python/{Nom_Projet}"
    dossier = Path(Nom_dossier)          # remplace par ton chemin
    dossier.mkdir(parents=True, exist_ok=True)


    # Chargement dataset
    df = load_datasets_breast_cancer()

    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Normalisation des features (évite les overflow avec SGD)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    automl = all_autoML(Nom_dossier, X_train, X_test, y_train, y_test)
    automl.use_all()
