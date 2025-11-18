# src/feature_analysis/report.py

from __future__ import annotations

from typing import Any, Dict, Union, Sequence, Optional
from dataclasses import is_dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import FEAnalysisConfig
from .targets import analyze_targets
from .features import analyze_features
from .leakage import detect_leakage
from .printing import print_fe_report

from src.analyse.dataset.all import (
    DatasetContextForLLM,
    BasicDatasetStats,
    TargetSummaryForLLM,
    FeatureSummaryForLLM,
    FEDatasetSnapshotForLLM,
)

from src.analyse.helper.helper_json_safe import make_json_safe

# ----------------------------------------------------------------------
# Fonction demandée : prendre un report et le sauvegarder en JSON
# ----------------------------------------------------------------------
def save_report_to_json(
    report: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Sauvegarde le report complet en JSON.

    - Convertit les dataclasses (FEDatasetSnapshotForLLM, TargetSummaryForLLM,
      FeatureSummaryForLLM, etc.) en dict.
    - Convertit aussi les objets pandas / numpy en structures JSON-friendly.
    - Les champs non remplis des dataclasses restent présents (None -> null en JSON).
    """
    json_ready = make_json_safe(report)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=2)
