from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Sequence
import json
import textwrap

import pandas as pd

from src.analyse.statistiques.report import FEAnalysisConfig, analyze_dataset_for_fe


# ----------------------------------------------------------------------
# 1) Dataclasses : représentation structurée du plan proposé par le LLM
# ----------------------------------------------------------------------


@dataclass
class FeatureTransformationSpec:
    """
    Spécifie une transformation de feature proposée par le LLM.

    Exemples :
      - type = "numeric_derived", transformation = "x1 / x2"
      - type = "categorical_encoding", encoding = "target_encoding"
      - type = "text_embedding", model = "sentence-transformers/all-MiniLM-L6-v2"
    """
    name: str
    type: str
    inputs: List[str] = field(default_factory=list)
    transformation: Optional[str] = None
    encoding: Optional[str] = None
    model: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class LLMFEPlan:
    """
    Plan complet de feature engineering renvoyé par le LLM.

    - features_plan : liste de features dérivées / encodages à créer
    - global_notes : recommandations globales sur le FE / le problème
    - questions_for_user : questions à poser à l'humain pour affiner le FE
    """
    features_plan: List[FeatureTransformationSpec] = field(default_factory=list)
    global_notes: List[str] = field(default_factory=list)
    questions_for_user: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None  # en cas de debug


# ----------------------------------------------------------------------
# 2) Pipeline LLM : analyse -> prompt -> LLM -> plan FE
# ----------------------------------------------------------------------


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
        max_examples_per_target: int = 5,
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
        self.max_examples_per_target = max_examples_per_target

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def analyse_and_plan(
        self,
        df: pd.DataFrame,
        target_cols: Sequence[str],
        llm_func: Callable[[str], str],
        *,
        user_description: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        print_prompt: bool = False,
    ) -> Dict[str, Any]:
        """
        Étape complète :
          - analyse du dataset
          - construction du prompt
          - appel au LLM
          - parsing du plan

        Parameters
        ----------
        df : pd.DataFrame
            Données complètes (features + cibles).
        target_cols : Sequence[str]
            Nom(s) des colonnes cibles.
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
        # 1) Analyse statistique
        report = analyze_dataset_for_fe(
            df,
            target_cols=target_cols,
            config=self.config,
            print_report=False,
        )

        # 2) Construire le prompt pour le LLM
        prompt = self._build_prompt_from_report(
            report,
            user_description=user_description,
            extra_instructions=extra_instructions,
        )

        if print_prompt:
            print("=" * 80)
            print("PROMPT ENVOYÉ AU LLM")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        # 3) Appel au LLM
        raw_response = llm_func(prompt)

        # 4) Parsing de la réponse en plan structuré
        plan = self._parse_llm_response(raw_response)
        plan.raw_response = raw_response  # pour debug

        return {
            "report": report,
            "prompt": prompt,
            "plan": plan,
        }

    # ------------------------------------------------------------------
    # 3) Construction du prompt à partir du rapport
    # ------------------------------------------------------------------

    def _build_prompt_from_report(
        self,
        report: Dict[str, Any],
        *,
        user_description: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> str:
        """
        Construit un prompt structuré pour le LLM à partir du rapport FE.
        On injecte les infos statistiques les plus importantes :
          - type de problème (classification / régression)
          - distribution de la cible
          - rôle, NaN, cardinalité, flags pour chaque feature
        """
        g = report["global"]
        targets = report["targets"]
        features = report["features"]

        # On limite le nombre de features détaillées pour ne pas exploser le contexte
        all_feature_names = list(features.keys())
        selected_feature_names = all_feature_names[: self.max_features_in_prompt]

        # --- Résumé des cibles ---
        targets_summary_lines: List[str] = []
        for t_name, t_info in targets.items():
            line = (
                f"- {t_name}: type={t_info['target_type']}, "
                f"problem={t_info['problem_hint']}, "
                f"n_unique={t_info['n_unique']}, "
                f"missing_rate={t_info['missing_rate']:.1%}"
            )
            targets_summary_lines.append(line)

        # --- Détail des features (sélectionnées) ---
        features_summary_lines: List[str] = []
        for fname in selected_feature_names:
            finfo = features[fname]
            flags = []
            if finfo["is_constant"]:
                flags.append("CONST")
            if finfo["is_id_like"]:
                flags.append("ID_LIKE")
            if finfo["high_cardinality"]:
                flags.append("HIGH_CARD")

            flags_str = f" flags={','.join(flags)}" if flags else ""
            line = (
                f"- {fname}: role={finfo['role']}, dtype={finfo['dtype']}, "
                f"n_unique={finfo['n_unique']}, "
                f"unique_ratio={finfo['unique_ratio']:.1%}, "
                f"missing_rate={finfo['missing_rate']:.1%}{flags_str}"
            )
            features_summary_lines.append(line)

        # --- Instructions sur le format de sortie JSON ---
        json_spec = textwrap.dedent(
            """
            Tu dois répondre UNIQUEMENT avec un JSON valide, sans texte autour.
            Le JSON doit avoir la structure suivante :

            {
              "features_plan": [
                {
                  "name": "nom_de_la_feature_creee_ou_modifiee",
                  "type": "numeric_derived | categorical_encoding | text_embedding | datetime_derived | other",
                  "inputs": ["col1", "col2"],
                  "transformation": "description_symbolique_de_la_transformation_ou_formule",
                  "encoding": "one_hot | target_encoding | ordinal | hashing | None",
                  "model": "nom_du_modele_ou_de_la_technique_si_applicable (ex: 'sentence-transformers/all-MiniLM-L6-v2')",
                  "reason": "raison_métier_ou_statistique_qui_explique_ce_choix"
                }
              ],
              "global_notes": [
                "recommandation_globale_1",
                "recommandation_globale_2"
              ],
              "questions_for_user": [
                "question_1",
                "question_2"
              ]
            }

            - "features_plan" : liste de transformations de features concrètes à créer
              ou d'encodages à appliquer.
            - "global_notes" : remarques générales sur le feature engineering à appliquer.
            - "questions_for_user" : questions à poser à l'utilisateur pour affiner encore
              le choix des features (ex: sémantique métier, contraintes, etc.).
            """
        )

        # --- Instructions sur le rôle du LLM ---
        system_instructions = textwrap.dedent(
            """
            Tu es un assistant expert en machine learning et en feature engineering.
            On te fournit un résumé statistique d'un dataset (cibles + features).
            Ton objectif est de proposer un plan de feature engineering intelligent
            qui exploite ces informations statistiques.

            Tu dois :
            - Utiliser le type de problème (classification / régression) pour adapter
              tes suggestions (encodage de la cible, gestion de l'imbalance, etc.).
            - Utiliser le rôle des features (numeric / categorical / text / datetime / boolean)
              pour proposer des transformations adaptées.
            - Tenir compte des taux de valeurs manquantes, de la cardinalité, et des flags
              (ID_LIKE, HIGH_CARD, CONST) pour éviter les fuites de données et le sur-apprentissage.
            - Proposer des features dérivées métierment pertinentes lorsque c'est possible
              (ratios, agrégations temporelles, etc.).
            - Proposer des encodages adaptés aux catégorielles (one-hot vs target encoding, etc.).
            - Proposer des représentations pour le texte (TF-IDF, embeddings, etc.) si pertinent.
            """
        )

        if extra_instructions:
            system_instructions += "\nInstructions supplémentaires :\n" + extra_instructions.strip()

        # --- Description utilisateur (optionnelle) ---
        user_desc_block = ""
        if user_description:
            user_desc_block = "Description donnée par l'utilisateur :\n" + user_description.strip() + "\n"

        # --- Assemblage final du prompt ---
        prompt = textwrap.dedent(
            f"""
            {system_instructions}

            ========================
            CONTEXTE DU DATASET
            ========================

            Nombre de lignes : {g['n_rows']}
            Nombre de features : {g['n_features']}
            Cibles : {g['target_cols']}

            {user_desc_block}

            ------------------------
            ANALYSE DES CIBLES
            ------------------------
            {chr(10).join(targets_summary_lines)}

            ------------------------
            ANALYSE DES FEATURES (sélection)
            ------------------------
            (Maximum de {self.max_features_in_prompt} features listées)

            {chr(10).join(features_summary_lines)}

            ========================
            FORMAT DE RÉPONSE ATTENDU
            ========================
            {json_spec}

            Rappels importants :
            - Ne propose pas de features qui utilisent directement la cible.
            - Ne propose pas d'utiliser des variables ID_LIKE comme features brutes.
            - Si une feature est de haute cardinalité, préfère des encodages compacts
              (target encoding, hashing, embeddings) plutôt que du one-hot naïf.
            - Tu peux utiliser les informations de rôle, de NaN, de cardinalité, et
              de flags pour justifier tes choix dans "reason".

            Réponds uniquement avec le JSON.
            """
        ).strip()

        return prompt

    # ------------------------------------------------------------------
    # 4) Parsing de la réponse du LLM
    # ------------------------------------------------------------------

    def _parse_llm_response(self, raw_response: str) -> LLMFEPlan:
        """
        Parse la réponse du LLM (texte) en LLMFEPlan.
        On attend un JSON selon la spec décrite dans le prompt.
        Si le parse échoue, on renvoie un plan vide avec la raw_response.
        """
        raw = raw_response.strip()

        # Parfois le LLM entoure le JSON de texte -> essayer d'extraire le bloc JSON
        # simple heuristique : chercher le premier '{' et le dernier '}'
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            raw_json = raw[start:end]
        except ValueError:
            # Pas de JSON détectable
            return LLMFEPlan(raw_response=raw_response)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # JSON mal formé
            return LLMFEPlan(raw_response=raw_response)

        plan = LLMFEPlan()

        # --- features_plan ---             
        features_plan = data.get("features_plan", [])
        if isinstance(features_plan, list):
            for item in features_plan:
                if not isinstance(item, dict):
                    continue
                spec = FeatureTransformationSpec(
                    name=item.get("name", ""),
                    type=item.get("type", "other"),
                    inputs=item.get("inputs", []) or [],
                    transformation=item.get("transformation"),
                    encoding=item.get("encoding"),
                    model=item.get("model"),
                    reason=item.get("reason"),
                )
                if spec.name:
                    plan.features_plan.append(spec)

        # --- global_notes ---
        global_notes = data.get("global_notes", [])
        if isinstance(global_notes, list):
            plan.global_notes = [str(x) for x in global_notes]

        # --- questions_for_user ---
        questions_for_user = data.get("questions_for_user", [])
        if isinstance(questions_for_user, list):
            plan.questions_for_user = [str(x) for x in questions_for_user]

        return plan
