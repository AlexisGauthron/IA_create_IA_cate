"""
Tests unitaires pour le module analyse.
"""
import sys
import os
import pytest

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ⚠️ IMPORTANT : Charger .env AVANT les décorateurs @skipif
from src.core.config import settings

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from src.core.io_utils import csv_to_dataframe_train_test


# Mapping des projets vers les noms de dossiers
PROJETS = {
    "titanic": {"label": "Survived", "path": "data/raw/titanic"},
    "verbatims": {"label": "Categorie", "path": "data/raw/verbatims"},
    "cate_metier": {"label": "label", "path": "data/raw/cate_metier"},
    "avis_client": {"label": "label", "path": "data/raw/avis_client"},
}


class TestDataLoading:
    """Tests pour le chargement des données."""

    def test_load_titanic_dataset(self):
        """Test le chargement du dataset Titanic."""
        df_train, df_test = csv_to_dataframe_train_test("data/raw/titanic")
        assert df_train is not None
        assert len(df_train) > 0
        assert "Survived" in df_train.columns

    def test_load_titanic_has_expected_columns(self):
        """Vérifie que le dataset Titanic a les colonnes attendues."""
        df_train, _ = csv_to_dataframe_train_test("data/raw/titanic")
        expected_cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age"]
        for col in expected_cols:
            assert col in df_train.columns, f"Colonne {col} manquante"


class TestAnalyse:
    """Tests pour le module d'analyse."""

    def test_print_api_key(self):
        """Affiche la clé API pour debug."""
        # Charger depuis le fichier .env
        from src.core.config import settings

        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        if api_key:
            # Afficher seulement les premiers et derniers caractères (sécurité)
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            print(f"\n[DEBUG] Clé API trouvée : {masked_key}")
            print(f"[DEBUG] Longueur : {len(api_key)} caractères")
        else:
            print("\n[DEBUG] Aucune clé API trouvée")
            print("[DEBUG] Vérifiez votre fichier .env ou la variable OPENAI_API_KEY")

    @pytest.mark.skipif(
        not settings.is_configured("openai"),
        reason="OPENAI_API_KEY non définie - test skipped"
    )
    def test_analyse_titanic_stats_only(self):
        """Test l'analyse statistique du dataset Titanic (sans LLM)."""
        from src.analyse.analyse import analyse

        df_train, _ = csv_to_dataframe_train_test("data/raw/titanic")

        # only_stats=True ne retourne rien, on vérifie juste que ça ne crash pas
        analyse(
            df_train,
            target_cols="Survived",
            nom="titanic",
            print_json=False,
            only_stats=True
        )

        # Si on arrive ici, le test a réussi
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
