"""
Tests unitaires pour le wrapper TPOT AutoML.
"""

import os
import sys

import pytest

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Vérifier si tpot est installé
try:
    from tpot import TPOTClassifier

    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False


@pytest.mark.skipif(not TPOT_AVAILABLE, reason="tpot non installé")
class TestTPOT:
    """Tests pour TPOT AutoML."""

    def test_import_tpot_wrapper(self):
        """Test l'import du wrapper TPOT."""
        from src.automl.supervised.tpot_wrapper import autoMl_tpot

        assert autoMl_tpot is not None

    def test_tpot_training(self):
        """Test l'entraînement TPOT sur breast_cancer."""
        from pathlib import Path

        from sklearn.model_selection import train_test_split

        from src.automl.supervised.tpot_wrapper import autoMl_tpot
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

        Tpot = autoMl_tpot(Nom_dossier, X_train, X_test, y_train, y_test)
        Tpot.tpot1()
        Tpot.predict_test()
        Tpot.enregistrement_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
