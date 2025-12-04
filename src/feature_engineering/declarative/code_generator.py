# pipeline_generate_code.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.analyse.statistiques.report import FEAnalysisConfig
from src.features_engineering.LLM.code_fe.parsing import parse_llm_response
from src.features_engineering.LLM.code_fe.prompt import build_prompt_generate_code


class LLMFeatureEngineeringCodeGenerator:
    """
    Pipeline qui génère directement du code Python permettant d'exécuter
    les transformations de Feature Engineering décrites à partir d'un rapport
    statistique (stats).

    Étapes :
      1) Génération du prompt FE (comme ta pipeline existante).
      2) Construction d'un second prompt demandant DU CODE EXÉCUTABLE.
      3) Appel au LLM.
      4) Extraction du code dans le bloc ```python ... ```.
    """

    def __init__(
        self,
        config: FEAnalysisConfig | None = None,
    ) -> None:
        self.config = config or FEAnalysisConfig()

    def analyse_and_generate_code(
        self,
        stats: dict[str, Any],
        llm_func: Callable[[str], str],
        *,
        print_prompt: bool = False,
        local_llm=False,
        just_prompt=True,
    ) -> dict[str, Any]:
        """
        Analyse + Génération du code FE directement.
        """

        # Construire le prompt pour le LLM
        if local_llm == False:
            code_prompt = build_prompt_generate_code(
                stats,
            )

        if print_prompt:
            print("=" * 80)
            print("PROMPT ENVOYÉ AU LLM")
            print("=" * 80)
            print(code_prompt)
            print("=" * 80)

        if not just_prompt:
            # Appel au LLM
            llm_code_output = llm_func(code_prompt)

            print("\n\nBRUTE :\n\n", llm_code_output, "\n\n")
            # Parsing de la réponse en plan structuré

            code = parse_llm_response(llm_code_output)

            return {
                "report": stats,
                "prompt": llm_code_output,
                "code": code,
            }
        else:
            return {
                "report": stats,
                "prompt": llm_code_output,
                "code": None,
            }
