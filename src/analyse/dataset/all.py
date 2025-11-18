from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any


# ----------------------------
# Stats utilisées pour le LLM
# ----------------------------
from src.analyse.dataset.contextes import DatasetContextForLLM
from src.analyse.dataset.globale import BasicDatasetStats
from src.analyse.dataset.cibles import TargetSummaryForLLM
from src.analyse.dataset.features import FeatureSummaryForLLM
from src.analyse.dataset.leakage import LeakageSignalForLLM

@dataclass
class FEDatasetSnapshotForLLM:
    """
    Snapshot complet du dataset AVANT choix de FE,
    destiné à être sérialisé (dict/JSON) et envoyé à un LLM.
    """
    context: DatasetContextForLLM
    basic_stats: BasicDatasetStats
    target: TargetSummaryForLLM
    features: List[FeatureSummaryForLLM] = field(default_factory=list)

    # Détections de fuites potentielles
    leakage_signals: List[LeakageSignalForLLM] = field(default_factory=list)
    
    # Optionnel : rappels sur les seuils utilisés pour les flags
    analysis_config: Dict[str, Any] = field(default_factory=dict)
    # ex: {"max_unique_cat_low": 20, "high_cardinality_threshold": 50, "id_unique_ratio_threshold": 0.95}

    # Notes globales pour le LLM
    global_notes: List[str] = field(default_factory=list)

    def to_llm_payload(self) -> Dict[str, Any]:
        """
        Convertit en dict propre, prêt à être passé au LLM
        (éventuellement converti ensuite en JSON).
        """
        return {
            "context": self.context.__dict__,
            "basic_stats": self.basic_stats.__dict__,
            "target": self.target.__dict__,
            "features": [f.__dict__ for f in self.features],
            "analysis_config": self.analysis_config,
            "global_notes": self.global_notes,
        }
