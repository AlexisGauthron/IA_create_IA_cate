from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Sequence

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
    descriptions_transformations: Optional[str] = None
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

