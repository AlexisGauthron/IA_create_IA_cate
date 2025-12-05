# Module Models

> Documentation technique complète du module `src/models/`

---

## Vue d'Ensemble

Le module **models** fournit une **abstraction unifiée** pour tous les modèles ML du projet :
1. **Interface commune** : `BaseModel` que tous les modèles implémentent
2. **Registre centralisé** : Point d'accès unique via `get_model()`
3. **Évaluation unifiée** : `CrossValidator` pour validation croisée multi-modèle
4. **Métriques standardisées** : Support complet classification et régression

---

## Architecture du Module

```
src/models/
├── __init__.py              # Exports centralisés
├── base.py                  # Interface abstraite BaseModel
├── registry.py              # Registre des modèles
├── config.py                # Configurations prédéfinies
│
├── wrappers/                # Wrappers pour différents algos
│   ├── __init__.py
│   ├── xgboost_wrapper.py   # XGBoostModel
│   ├── lightgbm_wrapper.py  # LightGBMModel
│   ├── sklearn_wrapper.py   # RandomForest, DecisionTree, Logistic
│   └── catboost_wrapper.py  # CatBoostModel (optionnel)
│
└── evaluation/              # Module d'évaluation
    ├── __init__.py
    ├── metrics.py           # Métriques standardisées
    └── cross_validator.py   # Validation croisée unifiée
```

---

## Modèles Disponibles

| Nom | Classe | Bibliothèque | Classification | Régression |
|-----|--------|--------------|----------------|------------|
| `xgboost` | XGBoostModel | xgboost | XGBClassifier | XGBRegressor |
| `lightgbm` | LightGBMModel | lightgbm | LGBMClassifier | LGBMRegressor |
| `randomforest` | RandomForestModel | sklearn | RandomForestClassifier | RandomForestRegressor |
| `decisiontree` | DecisionTreeModel | sklearn | DecisionTreeClassifier | DecisionTreeRegressor |
| `logistic` | LogisticRegressionModel | sklearn | LogisticRegression | Ridge |
| `catboost` | CatBoostModel | catboost | CatBoostClassifier | CatBoostRegressor |

---

## Classes Principales

### 1. BaseModel (Interface Abstraite)

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Interface commune pour tous les modèles ML."""

    def __init__(
        self,
        is_regression: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        self.is_regression = is_regression
        self.random_state = random_state
        self.model = None

    # --- Méthodes abstraites (à implémenter) ---

    @abstractmethod
    def get_name(self) -> str:
        """Nom du modèle (ex: 'xgboost')."""

    @abstractmethod
    def create_model(self, **hp) -> Any:
        """Crée une instance du modèle avec hyperparamètres."""

    @abstractmethod
    def get_default_params(self) -> dict:
        """Hyperparamètres par défaut."""

    @abstractmethod
    def get_hp_space(self) -> dict:
        """Espace de recherche pour HPO."""

    # --- Méthodes concrètes (héritées) ---

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        """Entraîne le modèle."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédictions."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilités (classification)."""

    def clone(self) -> "BaseModel":
        """Copie non entraînée du modèle."""

    def get_feature_importance(self) -> np.ndarray | None:
        """Importance des features (si disponible)."""
```

### 2. Registry (Registre Centralisé)

```python
from src.models import get_model, get_models, list_models

# Récupérer un modèle
model = get_model("xgboost", is_regression=False)

# Récupérer plusieurs modèles
models = get_models(["xgboost", "lightgbm", "randomforest"], is_regression=False)

# Lister tous les modèles disponibles
print(list_models())  # ['xgboost', 'lightgbm', 'randomforest', ...]

# Vérifier disponibilité
from src.models import is_model_available
if is_model_available("catboost"):
    model = get_model("catboost")

# Enregistrer un modèle custom
from src.models import register_model
register_model("my_model", MyCustomModel)
```

**Listes prédéfinies :**

```python
from src.models.registry import FAST_MODELS, DEFAULT_MODELS

FAST_MODELS = ["decisiontree", "logistic"]       # Pour screening
DEFAULT_MODELS = ["xgboost", "lightgbm", "randomforest"]  # Défaut
```

### 3. CrossValidator (Validation Croisée)

```python
from src.models.evaluation import CrossValidator, CVResult

cv = CrossValidator(n_folds=5, shuffle=True, random_state=42)
```

#### Évaluation Simple

```python
result: CVResult = cv.evaluate(model, X, y, metric="f1")

print(f"Score: {result.mean:.4f} +/- {result.std:.4f}")
print(f"Scores par fold: {result.scores}")
```

#### Évaluation Multi-Modèle

```python
from src.models.evaluation import MultiModelCVResult

models = get_models(["xgboost", "lightgbm", "randomforest"])
result: MultiModelCVResult = cv.evaluate_multi_model(
    models, X, y,
    metric="f1",
    aggregation="mean"  # mean, min, max, median
)

print(f"Meilleur modèle: {result.best_model}")
print(f"Score agrégé: {result.aggregated_score:.4f}")
print(f"Score XGBoost: {result.results['xgboost'].mean:.4f}")
```

#### Évaluation Multi-Métrique Pondérée

```python
from src.models.evaluation import WeightedMetricResult

metrics_config = [
    {"name": "recall", "weight": 0.6},
    {"name": "precision", "weight": 0.3},
    {"name": "f1", "weight": 0.1}
]  # Total = 1.0

result: WeightedMetricResult = cv.evaluate_weighted_metrics(
    model, X, y, metrics_config
)

print(f"Score pondéré: {result.weighted_score:.4f}")
print(f"Recall: {result.scores['recall']:.4f}")
print(f"Precision: {result.scores['precision']:.4f}")
```

#### Évaluation Rapide (Screening)

```python
# Évaluation rapide avec 1 fold et sous-échantillonnage
score = cv.quick_evaluate(model, X, y, metric="f1", sample_size=1000)
```

### 4. Métriques

```python
from src.models.evaluation.metrics import (
    get_metric,
    get_scorer,
    get_default_metric,
    list_metrics,
    compute_all_metrics
)

# Récupérer une fonction de métrique
metric_fn = get_metric("f1", is_regression=False)
score = metric_fn(y_true, y_pred)

# Scorer compatible sklearn
scorer = get_scorer("auc", is_regression=False)

# Métrique par défaut
default = get_default_metric(is_regression=False)  # "f1"
default = get_default_metric(is_regression=True)   # "rmse"

# Lister les métriques disponibles
metrics = list_metrics(is_regression=False)
# ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'precision', 'recall', 'auc', 'logloss']

metrics = list_metrics(is_regression=True)
# ['rmse', 'mse', 'mae', 'r2', 'mape']

# Calculer toutes les métriques
all_scores = compute_all_metrics(y_true, y_pred, y_proba, is_regression=False)
```

---

## Configurations Prédéfinies

```python
from src.models.config import (
    ScreeningConfig,
    DefaultConfig,
    HPOConfig,
    LLMFEConfig,
    LandmarkingConfig,
    get_config
)

# Accès par preset
config = get_config("llmfe")
```

| Config | Modèles | Folds | Usage |
|--------|---------|-------|-------|
| `ScreeningConfig` | decisiontree, logistic | 3 | Évaluation rapide |
| `DefaultConfig` | xgboost, lightgbm, randomforest | 5 | Production |
| `HPOConfig` | xgboost, lightgbm | 5 | Optimisation hyperparamètres |
| `LLMFEConfig` | xgboost, lightgbm, randomforest | 4 | Évaluation features (LLMFE) |
| `LandmarkingConfig` | decisiontree, logistic | 2 | Meta-learning |

---

## Hyperparamètres par Défaut

### XGBoost

```python
{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1
}
```

### LightGBM

```python
{
    "n_estimators": 100,
    "max_depth": -1,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
```

### RandomForest

```python
{
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": "sqrt"
}
```

---

## Utilisation par les Autres Modules

### LLMFE (`src/feature_engineering/llmfe/`)

```python
from src.models import CrossValidator, get_models

def evaluate_features(X, y, model_names=["xgboost"], n_folds=4):
    """Évalue les features générées sur plusieurs modèles."""
    models = get_models(model_names, is_regression=False)
    cv = CrossValidator(n_folds=n_folds)
    result = cv.evaluate_multi_model(models, X, y, aggregation="mean")
    return result.aggregated_score
```

### Meta-Learning (futur)

```python
from src.models import get_models
from src.models.config import LandmarkingConfig

# Landmarking rapide pour méta-features
config = LandmarkingConfig()
models = get_models(config.models, is_regression=False)
cv = CrossValidator(n_folds=config.n_folds)

for model in models:
    score = cv.quick_evaluate(model, X, y, sample_size=config.sample_size)
    meta_features[f"landmark_{model.get_name()}"] = score
```

---

## Exemples d'Utilisation

### Exemple 1 : Évaluation Simple

```python
from src.models import get_model, CrossValidator

# Créer modèle et évaluateur
model = get_model("xgboost", is_regression=False)
cv = CrossValidator(n_folds=5)

# Évaluer
result = cv.evaluate(model, X_train, y_train, metric="f1")
print(f"F1: {result.mean:.4f} +/- {result.std:.4f}")
```

### Exemple 2 : Comparaison Multi-Modèle

```python
from src.models import get_models, CrossValidator

models = get_models(["xgboost", "lightgbm", "randomforest"])
cv = CrossValidator(n_folds=5)

result = cv.evaluate_multi_model(models, X, y, metric="auc", aggregation="mean")

print(f"Meilleur: {result.best_model} ({result.results[result.best_model].mean:.4f})")
print(f"Score agrégé: {result.aggregated_score:.4f}")
```

### Exemple 3 : Optimisation Recall vs Precision

```python
from src.models import get_model, CrossValidator

model = get_model("xgboost")
cv = CrossValidator(n_folds=5)

# Prioriser le recall (détection fraude, maladie)
config = [
    {"name": "recall", "weight": 0.7},
    {"name": "precision", "weight": 0.3}
]

result = cv.evaluate_weighted_metrics(model, X, y, config)
print(f"Score optimisé: {result.weighted_score:.4f}")
```

---

## Points Clés d'Architecture

1. **Abstraction uniforme** : Tous les modèles implémentent `BaseModel`
2. **Registre centralisé** : `get_model()` comme point d'accès unique
3. **StratifiedKFold** : Préserve la distribution des classes
4. **Reproductibilité** : `random_state` partout
5. **Parallélisation** : `n_jobs=-1` par défaut
6. **Optionalité** : CatBoost optionnel (try/except ImportError)

---

## Voir Aussi

- [OVERVIEW.md](../architecture/OVERVIEW.md) - Vue d'ensemble du projet
- [MODULE_DEPENDENCIES.md](../architecture/MODULE_DEPENDENCIES.md) - Dépendances entre modules
- [core.md](./core.md) - Module core (config, LLM client)
