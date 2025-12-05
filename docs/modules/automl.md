# Module AutoML

> Documentation technique complète du module `src/automl/`

---

## Vue d'Ensemble

Le module **automl** orchestre **4 frameworks AutoML** en parallèle pour entraîner et comparer des modèles ML :

1. **FLAML** : Fast and Lightweight AutoML
2. **AutoGluon** : TabularPredictor d'Amazon
3. **TPOT** : Tree-based Pipeline Optimization Tool
4. **H2O** : H2O AutoML (le plus complet)

---

## Architecture du Module

```
src/automl/
├── __init__.py
├── path_config.py              # Gestion centralisée des chemins
├── runner.py                   # Orchestrateur (AutoMLRunner)
└── supervised/                 # Wrappers pour chaque framework
    ├── __init__.py
    ├── flaml_wrapper.py        # FlamlWrapper
    ├── autogluon_wrapper.py    # AutoGluonWrapper
    ├── h2o_wrapper.py          # H2OWrapper (1000+ lignes)
    └── tpot_wrapper.py         # TPOTWrapper
```

---

## Structure des Outputs

```
outputs/{project_name}/automl/
├── flaml/
│   └── time_budget_{N}/
│       └── model.pkl
│
├── autogluon/
│   └── time_budget_{N}/
│       └── predictor/
│
├── tpot/
│   └── time_budget_{N}/
│       └── pipeline.pkl
│
├── h2o/
│   └── time_budget_{N}/
│       ├── best_model/
│       │   ├── model_raw.json
│       │   ├── variable_importance.csv
│       │   └── MOJO/
│       ├── leaderboard.csv
│       └── all_models/
│           ├── XGBoost_1_AutoML/
│           ├── GBM_1_AutoML/
│           └── StackedEnsemble_AllModels/
│               └── base_models/
│
├── results/
│   ├── comparison.json
│   └── leaderboard.csv
│
└── logs/
    └── automl.log
```

---

## AutoMLRunner (Orchestrateur)

### Initialisation

```python
from src.automl.runner import AutoMLRunner

runner = AutoMLRunner(
    output_dir="outputs/titanic",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
```

### Lancer Tous les Frameworks

```python
runner.use_all(model=["autogluon", "flaml", "tpot", "h2o"])

# Afficher les résultats
runner.compare_all_predict(model=["autogluon", "flaml", "tpot", "h2o"])
```

### Lancer un Framework Spécifique

```python
# FLAML
runner.flaml()
print(f"Score FLAML: {runner.score_flaml}")

# AutoGluon
runner.autogluon()
print(f"Score AutoGluon: {runner.score_autogluon}")

# TPOT
runner.tpot()
print(f"Score TPOT: {runner.score_tpot}")

# H2O (avec budget temps)
runner.h2o(time_budget=300)  # 5 minutes
print(f"Score H2O: {runner.score_h2o}")
```

### Attributs de Résultats

```python
runner.score_flaml        # Score FLAML (F1)
runner.score_autogluon    # Score AutoGluon
runner.score_tpot         # Score TPOT
runner.score_h2o          # Score H2O
runner.errors             # Dict des erreurs rencontrées
```

---

## Wrappers de Frameworks

### FlamlWrapper

```python
from src.automl.supervised.flaml_wrapper import FlamlWrapper

wrapper = FlamlWrapper(X_train, X_test, y_train, y_test, output_dir)
wrapper.flaml()                    # Entraînement
score = wrapper.predict_test()     # Évaluation
wrapper.enregistrement_model()     # Sauvegarde
```

**Particularités** :
- Détecte automatiquement binaire (f1) vs multiclasse (macro_f1)
- Léger et rapide

### AutoGluonWrapper

```python
from src.automl.supervised.autogluon_wrapper import AutoGluonWrapper

wrapper = AutoGluonWrapper(X_train, X_test, y_train, y_test, output_dir)
wrapper.autogluon()
score = wrapper.predict_test()
```

**Particularités** :
- Presets configurables (medium_quality, high_quality, etc.)
- Convertit automatiquement numpy → DataFrame

### H2OWrapper

```python
from src.automl.supervised.h2o_wrapper import H2OWrapper

wrapper = H2OWrapper(X_train, X_test, y_train, y_test, target_col, output_dir)

# Pipeline complet
wrapper.use_all(time_budget=300)

# Ou étape par étape
wrapper.init_cluster()
wrapper.start_automl(time_budget=300)
wrapper.analyser_modele()
wrapper.save_best_model()
wrapper.save_leaderboard()
wrapper.sauvegarder_features_tous_modeles()
score = wrapper.predict_test()
wrapper.shutdown()
```

**Particularités** :
- Export MOJO (format production-ready)
- Leaderboard complet avec tous les modèles
- Variable importance détaillée
- Sauvegarde features par modèle

### TPOTWrapper

```python
from src.automl.supervised.tpot_wrapper import TPOTWrapper

wrapper = TPOTWrapper(X_train, X_test, y_train, y_test, output_dir)
wrapper.tpot1()
score = wrapper.predict_test()
wrapper.enregistrement_model()
```

**Particularités** :
- Optimise un pipeline sklearn complet
- Utilise `fitted_pipeline_` (TPOT v1.x)

---

## AutoMLPathConfig (Gestion des Chemins)

```python
from src.automl.path_config import AutoMLPathConfig

paths = AutoMLPathConfig(project_name="titanic")

# Chemins
paths.get_framework_dir("h2o")         # → .../automl/h2o/
paths.get_model_path("flaml")          # → .../automl/flaml/model.pkl
paths.results_dir                       # → .../automl/results/
paths.comparison_path                   # → .../automl/results/comparison.json

# Sauvegarde
paths.save_comparison({
    "flaml": 0.85,
    "autogluon": 0.87,
    "h2o": 0.88,
    "tpot": 0.84
})
```

---

## Détection Automatique

### Métrique

```python
# Détection automatique selon le nombre de classes
if n_classes == 2:
    metric = "f1"
else:
    metric = "f1_macro"  # ou "macro_f1" pour FLAML
```

### Type de Problème

```python
# Via src/analyse/ ou détection directe
if y.nunique() <= 20:
    problem_type = "classification"
else:
    problem_type = "regression"
```

---

## Gestion des Erreurs

Le module est résilient : une erreur sur un framework ne bloque pas les autres.

```python
runner.use_all()  # Continue même si H2O échoue

# Consulter les erreurs
for framework, error in runner.errors.items():
    print(f"{framework}: {error}")
```

**Hints utiles** (H2O) :

| Erreur | Solution |
|--------|----------|
| `illegal in h2oautoml id` | Retirer les `/` du project_name |
| `java` | Installer JDK 8+ |
| `port already in use` | Fermer le cluster H2O existant |

---

## Flow de Données

```
┌─────────────────────────────────────────────────────────┐
│ Dataset initial (X_train, X_test, y_train, y_test)      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ AutoMLRunner.use_all()                                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│   │  FLAML   │  │AutoGluon │  │   TPOT   │  │  H2O   │  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│        │             │             │            │        │
│        ▼             ▼             ▼            ▼        │
│   score_flaml  score_autogluon  score_tpot  score_h2o   │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ compare_all_predict() → Résumé des scores                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Outputs                                                  │
│   - Modèles sauvegardés par framework                   │
│   - comparison.json (scores comparés)                   │
│   - leaderboard.csv (H2O)                               │
│   - automl.log (traces)                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Interactions avec Autres Modules

### Depuis src/pipeline/

```python
# pipeline_autoMl.py
from src.automl.runner import AutoMLRunner
from src.core.io_utils import csv_to_dataframe_train_test
from src.core.preprocessing import df_to_list_kaggle

# Charger données
df_train, df_test = csv_to_dataframe_train_test(data_path)
X_train, X_test, y_train = df_to_list_kaggle(df_train, df_test, target_col)

# Lancer AutoML
runner = AutoMLRunner(output_dir, X_train, X_test, y_train, y_test)
runner.use_all()
```

### Depuis src/front/

```python
# pipeline_streamlit.py affiche les résultats
st.metric("FLAML", runner.score_flaml)
st.metric("H2O", runner.score_h2o)
```

---

## Exemple Complet

```python
from src.automl.runner import AutoMLRunner
from src.automl.path_config import AutoMLPathConfig
from src.core.io_utils import csv_to_dataframe_train_test
from src.core.preprocessing import df_to_list_kaggle

# 1. Charger les données
df_train, df_test = csv_to_dataframe_train_test("data/raw/titanic")
X_train, X_test, y_train = df_to_list_kaggle(df_train, df_test, "Survived")

# 2. Créer le runner
runner = AutoMLRunner(
    output_dir="outputs/titanic/automl",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=None,  # Pas de labels test (Kaggle)
)

# 3. Lancer tous les frameworks
runner.use_all(model=["flaml", "autogluon", "h2o"])

# 4. Comparer les scores
runner.compare_all_predict(model=["flaml", "autogluon", "h2o"])

# 5. Récupérer le meilleur
best = max(
    [("flaml", runner.score_flaml),
     ("autogluon", runner.score_autogluon),
     ("h2o", runner.score_h2o)],
    key=lambda x: x[1] or 0
)
print(f"Meilleur framework: {best[0]} ({best[1]:.4f})")
```

---

## Comparaison des Frameworks

| Framework | Vitesse | Précision | Features | Production |
|-----------|---------|-----------|----------|------------|
| **FLAML** | ⚡⚡⚡ Très rapide | ⭐⭐ Bonne | Léger, simple | joblib |
| **AutoGluon** | ⚡⚡ Rapide | ⭐⭐⭐ Très bonne | Presets, stacking | Custom |
| **TPOT** | ⚡ Lent | ⭐⭐ Bonne | Pipeline sklearn | sklearn |
| **H2O** | ⚡⚡ Moyen | ⭐⭐⭐ Excellente | MOJO, leaderboard | MOJO (Java) |

---

## Voir Aussi

- [OVERVIEW.md](../architecture/OVERVIEW.md) - Vue d'ensemble du projet
- [MODULE_DEPENDENCIES.md](../architecture/MODULE_DEPENDENCIES.md) - Dépendances
- [feature_engineering.md](./feature_engineering.md) - Module FE (étape précédente)
- [pipeline.md](./pipeline.md) - Module pipeline (orchestration)
