# Module Feature Engineering

Génération intelligente de features via LLM, DFS, et transformations.

📚 **Documentation complète** : [docs/modules/feature_engineering.md](../../docs/modules/feature_engineering.md)

## Sous-modules

| Dossier | Rôle |
|---------|------|
| `llmfe/` | LLM-based Feature Engineering (principal) |
| `dfs/` | Deep Feature Synthesis (featuretools) |
| `transforms/` | Transformations réutilisables |
| `declarative/` | Plans FE déclaratifs |
| `libs/` | Wrappers bibliothèques externes |
| `hybrid/` | Combinaison LLMFE + DFS |

## Fichiers Principaux

| Fichier | Rôle |
|---------|------|
| `path_config.py` | Gestion centralisée des chemins |
| `llmfe/llmfe_runner.py` | Point d'entrée LLMFE (`LLMFERunner`) |
| `llmfe/config.py` | Configuration (EvaluationConfig) |
| `llmfe/model_evaluator.py` | Évaluation multi-modèle |

## Usage Rapide

```python
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner

runner = LLMFERunner(project_name="titanic")
result = runner.run(
    df_train=df,
    target_col="Survived",
    max_samples=20,
    eval_models=["xgboost", "lightgbm"],
    eval_metric="f1",
)
print(f"Best score: {result['best_score']:.4f}")
```

## Évaluation Multi-Modèle

```python
# Multi-modèle
eval_models=["xgboost", "lightgbm", "randomforest"]
eval_aggregation="mean"

# Multi-métrique pondérée
eval_metrics_config=[
    {"name": "recall", "weight": 0.6},
    {"name": "precision", "weight": 0.4}
]
```
