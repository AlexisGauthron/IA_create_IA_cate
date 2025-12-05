# Module Models

Abstraction unifiée pour modèles ML : wrappers, registre, évaluation.

📚 **Documentation complète** : [docs/modules/models.md](../../docs/modules/models.md)

## Fichiers Principaux

| Fichier | Rôle |
|---------|------|
| `base.py` | Interface abstraite `BaseModel` |
| `registry.py` | Registre centralisé `get_model()` |
| `config.py` | Configurations prédéfinies (LLMFE, HPO, etc.) |

## Sous-dossiers

| Dossier | Contenu |
|---------|---------|
| `wrappers/` | XGBoost, LightGBM, RandomForest, etc. |
| `evaluation/` | CrossValidator, métriques |

## Usage Rapide

```python
from src.models import get_model, CrossValidator

model = get_model("xgboost", is_regression=False)
cv = CrossValidator(n_folds=5)
result = cv.evaluate(model, X, y, metric="f1")
print(f"F1: {result.mean:.4f}")
```

## Modèles Disponibles

`xgboost`, `lightgbm`, `randomforest`, `decisiontree`, `logistic`, `catboost`
