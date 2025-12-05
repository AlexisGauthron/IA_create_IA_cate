# Module Pipeline

Orchestrateur central : Analyse → Feature Engineering → AutoML.

📚 **Documentation complète** : [docs/modules/pipeline.md](../../docs/modules/pipeline.md)

## Fichiers

| Fichier | Rôle |
|---------|------|
| `pipeline_all.py` | Orchestrateur principal (`FullPipeline`) |
| `pipeline_autoMl.py` | Pipeline legacy (AutoML seul) |

## Usage Rapide

```python
from src.pipeline.pipeline_all import run_pipeline

result = run_pipeline(
    project_name="titanic",
    df_train=df,
    target_col="Survived",
)

print(f"Best: {result.best_framework} ({result.best_score:.4f})")
```

## Flux d'Exécution

```
Analyse → Détection Params → Feature Engineering → AutoML → Résumé
```

## Contrôle des Étapes

```python
# Analyse seule
run_pipeline(..., enable_fe=False, enable_automl=False)

# Avec LLM métier
run_pipeline(..., analyse_only_stats=False)

# Overrides manuels
run_pipeline(..., override_metric="auc", override_max_samples=30)
```
