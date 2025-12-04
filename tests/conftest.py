"""
Fixtures pytest partagées pour les tests.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ajouter src au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_dataframe():
    """DataFrame d'exemple pour les tests."""
    return pd.DataFrame(
        {
            "feature_num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_cat": ["A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def project_root():
    """Racine du projet."""
    return PROJECT_ROOT


@pytest.fixture
def data_dir(project_root):
    """Dossier des données."""
    return project_root / "data" / "raw"


@pytest.fixture
def outputs_dir(project_root):
    """Dossier des sorties."""
    return project_root / "outputs"
