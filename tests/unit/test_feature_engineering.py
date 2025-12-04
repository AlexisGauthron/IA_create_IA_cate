"""
Tests unitaires pour le module feature_engineering.
"""

import os
import sys

import pytest

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class TestFeatureEngineeringImports:
    """Tests pour vérifier que les imports fonctionnent."""

    def test_import_planner(self):
        """Test l'import du planner."""
        from src.feature_engineering.declarative.planner import LLMFeatureEngineeringPipeline

        assert LLMFeatureEngineeringPipeline is not None

    def test_import_parsing(self):
        """Test l'import du parsing."""
        from src.feature_engineering.declarative.parsing import (
            FeatureTransformationSpec,
            LLMFEPlan,
            parse_llm_response,
        )

        assert parse_llm_response is not None
        assert LLMFEPlan is not None
        assert FeatureTransformationSpec is not None

    def test_import_prompt(self):
        """Test l'import du prompt builder."""
        from src.feature_engineering.declarative.prompt import build_prompt_from_report

        assert build_prompt_from_report is not None


class TestParsing:
    """Tests pour le parsing des réponses LLM."""

    def test_parse_valid_json(self):
        """Test le parsing d'un JSON valide."""
        from src.feature_engineering.declarative.parsing import parse_llm_response

        raw_response = """
        {
            "features_plan": [
                {
                    "name": "age_squared",
                    "type": "numeric_derived",
                    "inputs": ["Age"],
                    "transformation": "square(Age)",
                    "reason": "Non-linear relationship with target"
                }
            ],
            "global_notes": ["Dataset has missing values in Age"],
            "questions_for_user": ["Is Age important for prediction?"]
        }
        """

        plan = parse_llm_response(raw_response)

        assert len(plan.features_plan) == 1
        assert plan.features_plan[0].name == "age_squared"
        assert plan.features_plan[0].type == "numeric_derived"
        assert len(plan.global_notes) == 1
        assert len(plan.questions_for_user) == 1

    def test_parse_invalid_json(self):
        """Test le parsing d'un JSON invalide."""
        from src.feature_engineering.declarative.parsing import parse_llm_response

        raw_response = "This is not valid JSON"
        plan = parse_llm_response(raw_response)

        # Doit retourner un plan vide avec raw_response
        assert len(plan.features_plan) == 0
        assert plan.raw_response == raw_response

    def test_parse_json_with_text_around(self):
        """Test le parsing d'un JSON entouré de texte."""
        from src.feature_engineering.declarative.parsing import parse_llm_response

        raw_response = """
        Here is my analysis:
        {
            "features_plan": [],
            "global_notes": ["Test note"],
            "questions_for_user": []
        }
        That's my recommendation.
        """

        plan = parse_llm_response(raw_response)
        assert len(plan.global_notes) == 1
        assert plan.global_notes[0] == "Test note"


class TestLLMClient:
    """Tests pour le client LLM."""

    def test_import_llm_client(self):
        """Test l'import du client LLM."""
        from src.core.llm_client import OllamaClient

        assert OllamaClient is not None

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY non définie - test skipped"
    )
    def test_openai_client_init(self):
        """Test l'initialisation du client OpenAI."""
        from src.core.llm_client import OllamaClient

        client = OllamaClient(
            provider="openai",
            model="gpt-4o-mini",
        )
        assert client is not None
        assert client.provider == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
