# Module Feature Engineering

> Documentation technique complète du module `src/feature_engineering/`

---

## Vue d'Ensemble

Le module **feature_engineering** est le **cœur transformationnel** du pipeline. Il génère des features intelligentes via :

1. **LLMFE** : Feature Engineering guidé par LLM (génération itérative de code)
2. **DFS** : Deep Feature Synthesis (agrégations structurelles)
3. **Hybrid** : Combinaison LLMFE + DFS
4. **Declarative** : Parsing de plans de FE déclarés par LLM

---

## Architecture du Module

```
src/feature_engineering/
├── __init__.py
├── path_config.py                  # Gestion centralisée des chemins
│
├── llmfe/                          # LLM-based Feature Engineering
│   ├── llmfe_runner.py             # Point d'entrée (LLMFERunner)
│   ├── pipeline.py                 # Orchestration complète
│   ├── config.py                   # Config, EvaluationConfig
│   ├── buffer.py                   # ExperienceBuffer (population)
│   ├── sampler.py                  # Sampler (génération LLM)
│   ├── evaluator.py                # Évaluation code généré
│   ├── model_evaluator.py          # Évaluation multi-modèle
│   ├── feature_formatter.py        # Formatage features pour prompt
│   ├── feature_insights.py         # Insights depuis src/analyse/
│   ├── evolution_tracker.py        # Suivi évolution
│   └── prompts/                    # Templates de prompts
│
├── dfs/                            # Deep Feature Synthesis
│   ├── runner.py                   # DFSRunner
│   ├── config.py                   # DFSConfig
│   ├── selection.py                # FeatureSelector
│   └── primitives.py               # Primitives featuretools
│
├── declarative/                    # FE déclaratif (plans)
│   ├── parsing.py                  # Parse plans LLM
│   ├── planner.py                  # Génération de plans
│   └── prompt.py                   # Construction prompts
│
└── hybrid/                         # LLMFE + DFS combinés
    ├── runner.py                   # HybridFeatureEngineer
    └── config.py                   # Configuration hybrid
```

---

## Structure des Outputs

```
outputs/{project_name}/feature_engineering/
├── features/                   # DataFrames transformés
│   ├── train_fe.parquet
│   ├── test_fe.parquet
│   └── feature_columns.json
│
├── llmfe/                      # Résultats LLMFE
│   ├── samples/                # JSON samples générés
│   ├── results/                # best_model.json, all_scores.json
│   └── tensorboard/            # Logs TensorBoard
│
├── transforms/                 # Pipeline sauvegardé
│   └── pipeline.pkl
│
├── dataset_fe/                 # CSV exportés
│   ├── train_fe.csv
│   └── test_fe.csv
│
├── specs/                      # Spécifications
└── logs/
```

---

## LLMFE : LLM-based Feature Engineering

### Concept

LLMFE utilise un **algorithme évolutionnaire** où un LLM génère du code Python pour créer des features :

1. Le LLM reçoit un prompt avec les meilleures solutions actuelles
2. Il génère N nouvelles transformations
3. Chaque transformation est évaluée (validation croisée multi-modèle)
4. Les meilleures sont gardées et alimentent le prochain prompt
5. Répéter jusqu'à convergence ou max_iterations

### LLMFERunner (Point d'entrée)

```python
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner

runner = LLMFERunner(project_name="titanic")
result = runner.run(
    df_train=df,
    target_col="Survived",
    is_regression=False,

    # Itérations
    max_samples=20,

    # LLM
    use_api=True,
    api_model="gpt-4o",

    # Parallélisation
    num_samplers=1,
    num_evaluators=1,
    samples_per_prompt=3,

    # Évaluation MULTI-MODÈLE
    eval_models=["xgboost", "lightgbm", "randomforest"],
    eval_aggregation="mean",
    eval_metric="f1",

    # OU Évaluation MULTI-MÉTRIQUE PONDÉRÉE
    eval_metrics_config=[
        {"name": "recall", "weight": 0.6},
        {"name": "precision", "weight": 0.3},
        {"name": "f1", "weight": 0.1}
    ],

    # Insights (optionnel - auto-calculé sinon)
    analyse_path="outputs/titanic/analyse/stats/report_stats.json",
)

print(f"Best score: {result['best_score']:.4f}")
print(f"Features générées: {result['n_features_generated']}")
```

### Retour de LLMFERunner.run()

```python
{
    "path_config": FeatureEngineeringPathConfig,
    "project_dir": str,
    "results_dir": str,
    "samples_dir": str,
    "scores": list[float],
    "n_features_per_iteration": list[int],
    "n_features_generated": int,
    "best_score": float,
    "final_score": float,
    "n_iterations": int,
}
```

### EvaluationConfig

```python
from src.feature_engineering.llmfe.config import EvaluationConfig

# Configuration legacy (XGBoost seul)
EVAL_LEGACY = EvaluationConfig()

# Multi-modèle
EVAL_MULTI_MODEL = EvaluationConfig(
    model_names=("xgboost", "lightgbm", "randomforest"),
    aggregation="mean"
)

# Rapide (screening)
EVAL_FAST = EvaluationConfig(
    model_names=("decisiontree", "logistic"),
    n_folds=3
)

# Multi-métrique pondérée
EVAL_WEIGHTED = EvaluationConfig(
    model_names=("xgboost",),
    metrics_config=(
        {"name": "recall", "weight": 0.6},
        {"name": "precision", "weight": 0.4}
    )
)
```

### FeatureInsights (Source: src/analyse/)

Les insights sur les features proviennent du module `analyse` :

```python
from src.feature_engineering.llmfe.feature_insights import FeatureInsights

# Charger depuis rapport existant
insights = FeatureInsights.from_json("outputs/titanic/analyse/stats/report_stats.json")

# Ou lancer l'analyse automatiquement
insights = FeatureInsights.from_analyse(df, target_col="Survived", project_name="titanic")

# Contenu d'un FeatureInsight
insight = insights.get("Age")
print(f"Type: {insight.inferred_type}")  # numeric
print(f"Missing: {insight.missing_rate:.1%}")  # 19.9%
print(f"Corrélation: {insight.correlation:.3f}")  # -0.077
print(f"Hints FE: {insight.fe_hints}")  # ['numeric_imputation', 'candidate_for_scaling']
```

### Composants Internes

#### ExperienceBuffer

Population évolutionnaire avec **multi-îles** pour diversité :

```python
from src.feature_engineering.llmfe.buffer import ExperienceBuffer
from src.feature_engineering.llmfe.config import ExperienceBufferConfig

config = ExperienceBufferConfig(
    functions_per_prompt=2,    # Meilleures fonctions dans le prompt
    num_islands=3,             # Nombre d'îles (diversité)
    reset_period=4*60*60,      # Reset îles faibles (4h)
)
```

#### Sampler

Génère les continuations via LLM :

```python
# Utilise LocalLLM (Llama-3.1) ou APILLM (OpenAI)
# Boucle: get_prompt() → LLM → samples → evaluate
```

#### Evaluator

Exécute le code généré en sandbox et note :

```python
# 1. Nettoie le code (AST)
# 2. Exécute avec timeout
# 3. Évalue avec model_evaluator
# 4. Retourne score (< 0 = erreur, [0,1] = score)
```

---

## DFS : Deep Feature Synthesis

### Concept

DFS crée des features par **agrégations structurelles** :
- Groupe par colonnes catégorielles
- Applique des primitives (mean, sum, max, count, etc.)
- Génère des features comme `pclass_mean_age`, `sex_count_survived`

### DFSRunner

```python
from src.feature_engineering.dfs.runner import run_dfs
from src.feature_engineering.dfs.config import DFSConfig

config = DFSConfig(
    max_depth=2,
    max_features=100,
    aggregation_primitives=["mean", "sum", "max", "min", "count"],
    transformation_primitives=["year", "month", "weekday"],
)

result = run_dfs(
    df_train=df,
    target_col="Survived",
    project_name="titanic",
    config=config,
)

print(f"Features générées: {result.n_features_generated}")
print(f"Score final: {result.final_score:.4f}")
```

---

## Declarative : Plans FE

### Concept

Au lieu de générer du code, le LLM **déclare un plan** de transformations :

```python
from src.feature_engineering.declarative.parsing import parse_llm_response, LLMFEPlan

plan: LLMFEPlan = parse_llm_response(llm_json_response)

for spec in plan.features_plan:
    print(f"Feature: {spec.name}")
    print(f"Type: {spec.type}")  # numeric_derived, categorical_encoding
    print(f"Inputs: {spec.inputs}")
    print(f"Transformation: {spec.transformation}")  # "x1 / x2"
```

### FeatureTransformationSpec

```python
@dataclass
class FeatureTransformationSpec:
    name: str                   # Nom de la nouvelle feature
    type: str                   # numeric_derived, categorical_encoding, text_embedding
    inputs: list[str]           # Colonnes source
    transformation: str | None  # Formule ("x1 / x2")
    encoding: str | None        # "target_encoding"
    model: str | None           # "sentence-transformers/..."
    reason: str | None          # Justification métier
```

---

## Hybrid : LLMFE + DFS Combinés

```python
from src.feature_engineering.hybrid.runner import HybridFeatureEngineer
from src.feature_engineering.hybrid.config import HybridConfig

config = HybridConfig(
    llmfe_max_iterations=15,
    dfs_config="synthetic_exhaustive",
    max_features=75,
)

engineer = HybridFeatureEngineer(project_name="titanic", config=config)
result = engineer.run(train_df, "Survived")

# Données transformées
df_transformed = engineer.get_transformed_data()
```

**Configurations prédéfinies** : `default`, `fast`, `exhaustive`, `llmfe_only`, `dfs_only`

---

## Gestion des Chemins

```python
from src.feature_engineering.path_config import FeatureEngineeringPathConfig

paths = FeatureEngineeringPathConfig(project_name="titanic")

# Chemins
paths.features_dir              # → outputs/titanic/feature_engineering/features/
paths.llmfe_dir                 # → outputs/titanic/feature_engineering/llmfe/
paths.train_features_path       # → .../features/train_fe.parquet
paths.llmfe_best_model_path     # → .../llmfe/results/best_model.json

# Sauvegarde
paths.save_train_features(df)
paths.save_test_features(df)
paths.save_llmfe_sample(order, code, score)
paths.save_llmfe_best_model(model_info)

# Lecture
df = pd.read_parquet(paths.train_features_path)
```

---

## Interactions avec Autres Modules

### Depuis src/analyse/

```python
# FeatureInsights charge les insights depuis le rapport d'analyse
insights = FeatureInsights.from_json(
    "outputs/titanic/analyse/stats/report_stats.json"
)

# Utilisé pour :
# - Enrichir le prompt LLM avec les stats des features
# - Indiquer les corrélations avec la cible
# - Signaler les flags (HIGH_CARDINALITY, ID_LIKE, etc.)
```

### Vers src/models/

```python
# model_evaluator.py utilise CrossValidator pour évaluer
from src.models import CrossValidator, get_models

models = get_models(["xgboost", "lightgbm"], is_regression=False)
cv = CrossValidator(n_folds=4)
result = cv.evaluate_multi_model(models, X_transformed, y)
```

### Vers src/automl/ et src/pipeline/

```python
# Les features générées sont sauvegardées en parquet
# AutoML les charge pour entraîner les modèles finaux
train_fe = pd.read_parquet(paths.train_features_path)
```

---

## Flow de Données Complet

```
ENTRÉE: df_train, target_col
         │
         ▼
┌────────────────────────────────────────────────────┐
│ 1. Charger/créer FeatureInsights (src/analyse/)    │
└────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│ 2. LLMFE Pipeline                                   │
│   ┌─────────────────────────────────────┐          │
│   │ ExperienceBuffer (population)       │          │
│   │   ↓                                 │          │
│   │ Sampler (prompt → LLM → code)       │          │
│   │   ↓                                 │          │
│   │ Evaluator (exécute + évalue)        │──┐       │
│   │   ↓                                 │  │       │
│   │ Score → Buffer (évolution)          │  │       │
│   │   ↓                                 │  │       │
│   │ Répéter N fois                      │  │       │
│   └─────────────────────────────────────┘  │       │
│                                            │       │
│   model_evaluator ←───────────────────────┘       │
│   (CrossValidator multi-modèle)                    │
└────────────────────────────────────────────────────┘
         │
         ▼ (optionnel)
┌────────────────────────────────────────────────────┐
│ 3. DFS (agrégations structurelles)                  │
└────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│ 4. Sauvegarde                                       │
│   - features/train_fe.parquet                      │
│   - llmfe/results/best_model.json                  │
│   - llmfe/samples/*.json                           │
└────────────────────────────────────────────────────┘
         │
         ▼
SORTIE: df_train_fe (features enrichies)
        best_score, n_iterations, etc.
```

---

## Exemple Complet

```python
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner
from src.feature_engineering.path_config import FeatureEngineeringPathConfig

# 1. Créer le runner
runner = LLMFERunner(project_name="titanic")

# 2. Lancer LLMFE
result = runner.run(
    df_train=train_df,
    target_col="Survived",
    max_samples=20,
    use_api=True,
    api_model="gpt-4o",
    eval_models=["xgboost", "lightgbm", "randomforest"],
    eval_metric="f1",
)

# 3. Récupérer les chemins
paths = result["path_config"]

# 4. Charger les features transformées
import pandas as pd
train_fe = pd.read_parquet(paths.train_features_path)

print(f"Colonnes originales: {len(train_df.columns)}")
print(f"Colonnes après FE: {len(train_fe.columns)}")
print(f"Best score: {result['best_score']:.4f}")
```

---

## Voir Aussi

- [OVERVIEW.md](../architecture/OVERVIEW.md) - Vue d'ensemble du projet
- [MODULE_DEPENDENCIES.md](../architecture/MODULE_DEPENDENCIES.md) - Dépendances
- [models.md](./models.md) - Module models (évaluation)
- [analyse.md](./analyse.md) - Module analyse (insights)
