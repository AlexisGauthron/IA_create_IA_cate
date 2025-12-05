# src/feature_engineering/hybrid/__init__.py
"""
Module de Feature Engineering Hybride (LLMFE + DFS).

Ce module combine deux approches complémentaires :
- LLMFE : Features métier intelligentes via LLM (prioritaire)
- DFS : Features structurelles via Deep Feature Synthesis

Exemple d'utilisation:
```python
from src.feature_engineering.hybrid import (
    HybridFeatureEngineer,
    HybridConfig,
    run_hybrid_fe,
)

# Méthode 1: Fonction raccourcie
df_transformed, result = run_hybrid_fe(
    df_train=train_df,
    target_col="Survived",
    project_name="titanic",
)

# Méthode 2: Avec configuration personnalisée
config = HybridConfig(
    llmfe_max_iterations=15,
    dfs_config="synthetic_exhaustive",
    max_features=75,
)
engineer = HybridFeatureEngineer(project_name="titanic", config=config)
result = engineer.run(train_df, "Survived")
df_transformed = engineer.get_transformed_data()
```

Configurations prédéfinies:
- "default" : Configuration équilibrée
- "fast" : Moins d'itérations, plus rapide
- "exhaustive" : Maximum de features
- "llmfe_only" : LLMFE seul (pas de DFS)
- "dfs_only" : DFS seul (pas de LLM requis)
"""

from src.feature_engineering.hybrid.config import (
    HYBRID_CONFIGS,
    HybridConfig,
    get_hybrid_config,
)
from src.feature_engineering.hybrid.runner import (
    HybridFeatureEngineer,
    HybridResult,
    run_hybrid_fe,
)

__all__ = [
    # Classes principales
    "HybridFeatureEngineer",
    "HybridConfig",
    "HybridResult",
    # Fonction raccourcie
    "run_hybrid_fe",
    # Configs prédéfinies
    "HYBRID_CONFIGS",
    "get_hybrid_config",
]
