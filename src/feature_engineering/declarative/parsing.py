from __future__ import annotations

# parsing.py
import json
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Dataclasses : représentation structurée du plan proposé par le LLM
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
    inputs: list[str] = field(default_factory=list)
    transformation: str | None = None
    descriptions_transformations: str | None = None
    encoding: str | None = None
    model: str | None = None
    reason: str | None = None


@dataclass
class LLMFEPlan:
    """
    Plan complet de feature engineering renvoyé par le LLM.

    - features_plan : liste de features dérivées / encodages à créer
    - global_notes : recommandations globales sur le FE / le problème
    - questions_for_user : questions à poser à l'humain pour affiner le FE
    """

    features_plan: list[FeatureTransformationSpec] = field(default_factory=list)
    global_notes: list[str] = field(default_factory=list)
    questions_for_user: list[str] = field(default_factory=list)
    raw_response: str | None = None  # en cas de debug


def parse_llm_response(raw_response: str) -> LLMFEPlan:
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
                descriptions_transformations=item.get("descriptions_transformations"),
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
