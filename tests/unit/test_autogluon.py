"""
Tests unitaires pour le wrapper AutoGluon AutoML.
"""

import os
import sys

import pytest

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Vérifier si autogluon est installé
try:
    from autogluon.tabular import TabularPredictor

    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


@pytest.mark.skipif(not AUTOGLUON_AVAILABLE, reason="autogluon non installé")
class TestAutoGluon:
    """Tests pour AutoGluon AutoML."""

    def test_import_autogluon_wrapper(self):
        """Test l'import du wrapper AutoGluon."""
        from src.automl.supervised.autogluon_wrapper import autoMl_autogluon

        assert autoMl_autogluon is not None

    def test_autogluon_training(self):
        """Test l'entraînement AutoGluon sur breast_cancer."""
        from pathlib import Path

        from sklearn.model_selection import train_test_split

        from src.automl.supervised.autogluon_wrapper import autoMl_autogluon
        from src.core.io_utils import load_datasets_breast_cancer

        Nom_Projet = "breast_cancer_test"
        Nom_dossier = f"Modeles/python/{Nom_Projet}"
        dossier = Path(Nom_dossier)
        dossier.mkdir(parents=True, exist_ok=True)

        df = load_datasets_breast_cancer()
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        Autogluon = autoMl_autogluon(Nom_dossier, X_train, X_test, y_train, y_test)
        Autogluon.autogluon(time_budget=30)
        Autogluon.predict_test()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
