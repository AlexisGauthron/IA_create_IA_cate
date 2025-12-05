"""
Tests unitaires pour le wrapper H2O AutoML.
"""

import os
import sys

import pytest

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Vérifier si h2o est installé
try:
    import h2o

    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False


@pytest.mark.skipif(not H2O_AVAILABLE, reason="h2o non installé")
class TestH2O:
    """Tests pour H2O AutoML."""

    def test_import_h2o_wrapper(self):
        """Test l'import du wrapper H2O."""
        from src.automl.supervised.h2o_wrapper import H2OWrapper

        assert H2OWrapper is not None

    def test_h2o_training(self):
        """Test l'entraînement H2O sur breast_cancer."""
        from pathlib import Path

        from sklearn.model_selection import train_test_split

        from src.automl.supervised.h2o_wrapper import H2OWrapper
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

        wrapper = H2OWrapper(output_dir, X_train, X_test, y_train, y_test)
        wrapper.use_all(time_budget=30)
        wrapper.predict_test()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
