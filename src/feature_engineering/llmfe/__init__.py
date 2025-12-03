# llmfe/__init__.py
"""
LLMFE - LLM-based Feature Engineering

Module pour l'optimisation automatique de features via LLM et algorithme évolutif.
"""

from src.feature_engineering.llmfe.path_config import LLMFEPathConfig
from src.feature_engineering.llmfe.config import Config, ClassConfig, ExperienceBufferConfig
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner, run_llmfe

__all__ = [
    # Configuration
    "LLMFEPathConfig",
    "Config",
    "ClassConfig",
    "ExperienceBufferConfig",
    # Runner
    "LLMFERunner",
    "run_llmfe",
]

__version__ = "1.0.0"
