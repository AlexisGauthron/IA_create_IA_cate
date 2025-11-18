# src/feature_analysis/config.py
from dataclasses import dataclass

@dataclass
class FEAnalysisConfig:
    max_classes_for_classif: int = 20
    max_unique_cat_low: int = 20
    high_cardinality_threshold: int = 50
    text_unique_ratio_threshold: float = 0.5
    id_unique_ratio_threshold: float = 0.9
    high_missing_threshold: float = 0.3
    strong_corr_threshold: float = 0.97
