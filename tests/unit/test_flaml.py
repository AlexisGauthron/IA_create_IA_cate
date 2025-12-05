"""
Tests unitaires pour le wrapper FLAML AutoML.
"""

import os
import sys

import pytest

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Vérifier si flaml est installé
try:
    from flaml import AutoML

    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False


@pytest.mark.skipif(not FLAML_AVAILABLE, reason="flaml non installé")
class TestFLAML:
    """Tests pour FLAML AutoML."""

    def test_import_flaml_wrapper(self):
        """Test l'import du wrapper FLAML."""
        from src.automl.supervised.flaml_wrapper import FlamlWrapper

        assert FlamlWrapper is not None

    def test_flaml_training(self):
        """Test l'entraînement FLAML sur breast_cancer."""
        from pathlib import Path

        from sklearn.model_selection import train_test_split

        from src.automl.supervised.flaml_wrapper import FlamlWrapper
        from src.core.io_utils import load_datasets_breast_cancer

        project_name = "breast_cancer_test"
        output_dir = f"Modeles/python/{project_name}"
        dossier = Path(output_dir)
        dossier.mkdir(parents=True, exist_ok=True)

        df = load_datasets_breast_cancer()
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        wrapper = FlamlWrapper(output_dir, X_train, X_test, y_train, y_test)
        wrapper.flaml(time_budget=30)
        wrapper.predict_test()
        wrapper.enregistrement_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
