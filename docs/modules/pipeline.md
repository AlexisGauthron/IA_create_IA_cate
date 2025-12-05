# Module Pipeline

> Documentation technique complète du module `src/pipeline/`

---

## Vue d'Ensemble

Le module **pipeline** est l'**orchestrateur central** qui coordonne les 3 étapes du système ML :

1. **Analyse** → Génère le JSON de référence (source unique de vérité)
2. **Feature Engineering** → LLMFE avec paramètres auto-détectés
3. **AutoML** → Entraîne et compare 4 frameworks

---

## Architecture du Module

```
src/pipeline/
├── __init__.py
├── pipeline_all.py         # Orchestrateur principal (FullPipeline)
└── pipeline_autoMl.py      # Pipeline legacy simplifié (AutoML seul)
```

---

## Flux d'Exécution

```
┌─────────────────────────────────────────────────────────────────┐
│                      FULL PIPELINE                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1: ANALYSE                                                 │
│   • Génère report_stats.json (stats, types, imbalance)          │
│   • [Optionnel] report_full.json (enrichissement LLM)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2: DÉTECTION PARAMÈTRES                                    │
│   • Extrait task_type, metric, feature_format                   │
│   • Inférence auto basée sur n_rows, n_features, imbalance      │
│   • Applique overrides manuels si spécifiés                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 3: FEATURE ENGINEERING (LLMFE)                             │
│   • Utilise feature_format et max_samples détectés              │
│   • Génère features via LLM                                     │
│   • Sauvegarde train_fe.parquet, test_fe.parquet                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 4: AUTOML                                                  │
│   • Lance FLAML, AutoGluon, TPOT, H2O                           │
│   • Compare les scores                                          │
│   • Sélectionne le meilleur framework                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 5: RÉSUMÉ                                                  │
│   • Génère pipeline_summary.json                                │
│   • Contient tous les résultats et paramètres                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Classes Principales

### FullPipeline (Orchestrateur)

```python
from src.pipeline.pipeline_all import FullPipeline

pipeline = FullPipeline(
    project_name="titanic",
    target_col="Survived",
    output_dir="outputs",

    # Contrôle des étapes
    enable_analyse=True,
    enable_fe=True,
    enable_automl=True,
    force_analyse=False,       # Force re-analyse même si JSON existe

    # Config analyse
    analyse_only_stats=True,   # Stats seul (pas LLM métier)
    analyse_provider="openai",
    analyse_model="gpt-4o-mini",
    with_correlations=False,

    # Config LLMFE
    llmfe_model="gpt-3.5-turbo",
    eval_metric="auto",
    eval_models=["xgboost", "lightgbm"],
    eval_aggregation="mean",

    # Config AutoML
    automl_frameworks=["flaml", "autogluon", "h2o", "tpot"],

    # Overrides manuels (optionnel)
    override_metric="f1",
    override_max_samples=20,
)

result = pipeline.run(df_train, df_test)
```

### run_pipeline (Fonction Raccourcie)

```python
from src.pipeline.pipeline_all import run_pipeline

result = run_pipeline(
    project_name="titanic",
    df_train=df,
    target_col="Survived",
    # ... mêmes paramètres que FullPipeline
)
```

### PipelineResult (Résultat)

```python
@dataclass
class PipelineResult:
    # Résultats des étapes
    analyse_result: dict | None
    detected_params: DetectedParams
    feature_engineering_result: dict
    automl_result: dict

    # DataFrames transformés
    df_train_fe: pd.DataFrame
    df_test_fe: pd.DataFrame

    # Meilleur modèle
    best_model: Any
    best_score: float
    best_framework: str

    # Métadonnées
    timestamp: str
    output_dir: Path
```

---

## Auto-Détection des Paramètres

### DetectedParams

Classe qui extrait les paramètres depuis le JSON d'analyse :

```python
@dataclass
class DetectedParams:
    target_col: str
    problem_type: str         # "binary_classification", "multiclass", "regression"
    task_type: str            # "classification" ou "regression"
    n_rows: int
    n_features: int
    n_text: int
    n_numeric: int
    n_categorical: int
    is_imbalanced: bool
    imbalance_ratio: float

    # Paramètres inférés
    metric: str               # Résolu depuis LLM ou suggestions
    feature_format: str       # "basic", "tags", "hierarchical"
    max_samples: int          # Itérations LLMFE
    time_budget: int          # Secondes pour AutoML
```

### InferenceConfig (Seuils)

```python
@dataclass
class InferenceConfig:
    # Format des features (complexité)
    format_basic_max_features: int = 5
    format_complexity_many_features: float = 0.4
    format_complexity_has_text: float = 0.2
    format_hierarchical_threshold: float = 0.5

    # Max samples LLMFE
    max_samples_small: int = 10      # n_features <= 10
    max_samples_medium: int = 15     # 10 < n_features <= 30
    max_samples_large: int = 25      # n_features > 30

    # Time budget AutoML
    time_budget_small: int = 60      # n_rows < 1000
    time_budget_medium: int = 120    # 1000 <= n_rows < 50000
    time_budget_large: int = 300     # n_rows >= 50000
```

### Logique d'Inférence

#### Metric (Priorité LLM > Suggestions)

```python
# 1. Chercher final_metric du LLM
if llm_analysis and llm_analysis.get("final_metric"):
    return llm_analysis["final_metric"]

# 2. Sinon, suggestions basées sur stats
if task_type == "classification":
    if is_imbalanced:
        return "f1"  # Ou auc
    else:
        return "accuracy"
else:
    return "rmse"
```

#### Feature Format

```python
score = 0.0

if n_features > 50:
    score += 0.4
elif n_features > 20:
    score += 0.2

if n_text > 0:
    score += 0.2

if missing_rate > 0.3:
    score += 0.2

if n_features <= 5:
    return "basic"
elif score > 0.5:
    return "hierarchical"
else:
    return "tags"
```

---

## Structure des Outputs

```
outputs/{project_name}/
├── analyse/
│   ├── stats/
│   │   └── report_stats.json      ← Source unique de vérité
│   ├── full/
│   │   └── report_full.json       ← Avec annotations LLM
│   └── agent_llm/
│       └── conversation.json
│
├── feature_engineering/
│   ├── features/
│   │   ├── train_fe.parquet
│   │   └── test_fe.parquet
│   └── llmfe/
│       ├── samples/
│       └── results/
│
├── models/
│   ├── flaml_model.pkl
│   ├── autogluon_model.pkl
│   ├── h2o_model.pkl
│   └── tpot_model.pkl
│
└── pipeline_summary.json          ← Résumé final
```

---

## Principe : Source Unique de Vérité

```
┌─────────────────────────────────────────────────────────────────┐
│ PRINCIPE FONDAMENTAL                                             │
│                                                                  │
│ src/analyse/ génère report_stats.json                           │
│        ↓                                                        │
│ LE PIPELINE LIT CE JSON                                         │
│        ↓                                                        │
│ Toutes les étapes utilisent ces paramètres                      │
│                                                                  │
│ AVANTAGES:                                                       │
│ • Cohérence garantie entre étapes                               │
│ • Reproductibilité                                              │
│ • Réutilisation d'analyses existantes                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Exemples d'Utilisation

### Pipeline Complet (Auto-détecté)

```python
from src.pipeline.pipeline_all import run_pipeline
import pandas as pd

df = pd.read_csv("data/raw/titanic/train.csv")

result = run_pipeline(
    project_name="titanic",
    df_train=df,
    target_col="Survived",
)

print(f"Task: {result.detected_params.task_type}")
print(f"Metric: {result.detected_params.metric}")
print(f"Best: {result.best_framework} ({result.best_score:.4f})")
```

### Analyse Seule

```python
result = run_pipeline(
    project_name="titanic",
    df_train=df,
    target_col="Survived",
    enable_fe=False,
    enable_automl=False,
)
```

### Avec Enrichissement LLM

```python
result = run_pipeline(
    project_name="titanic",
    df_train=df,
    target_col="Survived",
    analyse_only_stats=False,  # Active le chat LLM
    analyse_provider="openai",
    analyse_model="gpt-4o",
)
```

### Avec Overrides Manuels

```python
result = run_pipeline(
    project_name="titanic",
    df_train=df,
    target_col="Survived",
    override_metric="auc",
    override_max_samples=30,
    override_time_budget=600,
)
```

### Pipeline AutoML Seul (Legacy)

```python
from src.pipeline.pipeline_autoMl import pipeline_create_model

pipeline_create_model(
    project_name="titanic",
    target_col="Survived",
    data_dir="data/raw",
)
```

---

## Interactions avec Autres Modules

### Analyse (src/analyse/)

```python
from src.analyse.statistiques.report import analyze_dataset_for_fe
from src.analyse.metier.chatbot_llm import BusinessClarificationBot
from src.analyse.path_config import AnalysePathConfig
```

### Feature Engineering (src/feature_engineering/)

```python
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner
from src.feature_engineering.llmfe.feature_insights import FeatureInsights
from src.feature_engineering.path_config import FeatureEngineeringPathConfig
```

### AutoML (src/automl/)

```python
from src.automl.runner import AutoMLRunner
from src.automl.path_config import AutoMLPathConfig
```

### Core (src/core/)

```python
from src.core.llm_client import OllamaClient
from src.core.io_utils import csv_to_dataframe_train_test
from src.core.preprocessing import df_to_list_kaggle
```

---

## Voir Aussi

- [OVERVIEW.md](../architecture/OVERVIEW.md) - Vue d'ensemble du projet
- [MODULE_DEPENDENCIES.md](../architecture/MODULE_DEPENDENCIES.md) - Dépendances
- [analyse.md](./analyse.md) - Module analyse
- [feature_engineering.md](./feature_engineering.md) - Module FE
- [automl.md](./automl.md) - Module AutoML
