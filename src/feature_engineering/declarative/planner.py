# pipeline.py
from __future__ import annotations

from typing import Callable, Dict, Any, Optional
import textwrap

from src.analyse.statistiques.config import FEAnalysisConfig
from src.feature_engineering.declarative.prompt import build_prompt_from_report
from src.feature_engineering.declarative.parsing import parse_llm_response

class LLMFeatureEngineeringPipeline:
    """
    Pipeline qui :
      1) Analyse statistiquement un DataFrame (features + cibles)
         via analyze_dataset_for_fe.
      2) Construit un prompt structuré pour un LLM, en injectant :
         - informations sur la cible (classification / régression, imbalance…)
         - informations sur chaque feature (role, NaN, cardinalité, flags…)
      3) Appelle un LLM pour proposer un plan de FE structuré (JSON).
      4) Parse la réponse JSON en objet LLMFEPlan.

    Le LLM n'est PAS appelé directement ici (API spécifique),
    on passe une fonction `llm_func(prompt: str) -> str` au moment de l'appel.
    """

    def __init__(
        self,
        config: Optional[FEAnalysisConfig] = None,
        max_features_in_prompt: int = 50,

    ) -> None:
        """
        Parameters
        ----------
        config : FEAnalysisConfig, optional
            Configuration des seuils pour l'analyse statistique. Si None,
            on utilise FEAnalysisConfig() par défaut.
        max_features_in_prompt : int
            Nombre maximum de features détaillées à inclure dans le prompt
            (on peut en avoir plus dans le dataset, mais on tronque pour garder
             un prompt raisonnable).
        max_examples_per_target : int
            Nombre max de classes / modalités / stats détaillées à montrer
            pour chaque cible.
        """
        self.config = config or FEAnalysisConfig()
        self.max_features_in_prompt = max_features_in_prompt

    def analyse_and_plan(
        self,
        stats: Dict[str, Any],
        llm_func: Callable[[str], str],
        *,
        user_description: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        print_prompt: bool = False,
        local_llm = False,
        just_prompt = True,
    ) -> Dict[str, Any]:
        """
        Étape complète :
          - analyse du dataset
          - construction du prompt
          - appel au LLM
          - parsing du plan

        Parameters
        ----------
        stats : Dict[str, Any] 
            Rapport détaillé json (au format fourni : "context", "basic_stats", "target", "features")
        llm_func : Callable[[str], str]
            Fonction qui appelle le LLM et renvoie une réponse texte.
            Exemple possible : wrapper autour d'Ollama, OpenAI, etc.
        user_description : str, optional
            Description métier libre du dataset (par l'utilisateur).
            Ex: "Ce dataset contient des tickets clients avec leurs catégories."
        extra_instructions : str, optional
            Instructions supplémentaires pour le LLM (style, contraintes, etc.).
        print_prompt : bool
            Si True, affiche le prompt généré (debug).

        Returns
        -------
        result : dict
            {
              "report": rapport_statistique,
              "prompt": prompt_envoye_au_llm,
              "plan": LLMFEPlan,
            }
        """

        # Construire le prompt pour le LLM
        if local_llm == False:
            prompt = build_prompt_from_report(
                stats,
                user_description=user_description,
                extra_instructions=extra_instructions,
                max_features_in_prompt=self.max_features_in_prompt,
            )

        if print_prompt:
            print("=" * 80)
            print("PROMPT ENVOYÉ AU LLM")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        if not just_prompt:
            # Appel au LLM
            raw_response = llm_func(prompt)

            # Parsing de la réponse en plan structuré

            plan = parse_llm_response(raw_response)
            plan.raw_response = raw_response  # pour debug

            return {
                "report": stats,
                "prompt": prompt,
                "plan": plan,
            }
        else: 
            return {
                "report": stats,
                "prompt": prompt,
                "plan": None,
            }