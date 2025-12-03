# Guide des Tests

Documentation complète pour exécuter et comprendre les tests du projet **IA_create_IA_cate**.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Structure des tests](#structure-des-tests)
4. [Tests d'intégration](#tests-dintégration)
   - [Test Analyse](#test-analyse)
   - [Test LLMFE](#test-llmfe)
   - [Test Pipeline Complet](#test-pipeline-complet-full-pipeline)
   - [Test Pipeline AutoML (Legacy)](#test-pipeline-automl-legacy)
   - [Test AutoML individuel](#test-automl-individuel)
5. [Tests unitaires](#tests-unitaires)
6. [Outputs générés](#outputs-générés)
7. [Troubleshooting](#troubleshooting)

---

## Vue d'ensemble

Le projet utilise **pytest** pour les tests. Les tests sont organisés en deux catégories :

| Type | Dossier | Description |
|------|---------|-------------|
| **Unitaires** | `tests/unit/` | Tests isolés de chaque composant |
| **Intégration** | `tests/integration/` | Tests end-to-end des pipelines complets |

---

## Prérequis

### Environnement Conda

```bash
# Activer l'environnement
conda activate Ia_create_ia
```

### Variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
# Clés API (optionnelles selon les tests)
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...

# Configuration par défaut
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
```

### Données

Les datasets doivent être présents dans `data/raw/` :

```
data/raw/
├── titanic/
│   ├── train.csv
│   └── test.csv
├── verbatims/
│   ├── train.csv
│   └── test.csv
├── cate_metier/
└── avis_client/
```

---

## Structure des tests

```
tests/
├── conftest.py              # Fixtures partagées (sample_dataframe, project_root, etc.)
├── __init__.py
│
├── unit/                    # Tests unitaires
│   ├── test_analyse.py      # Tests du module analyse
│   ├── test_autogluon.py    # Tests AutoGluon
│   ├── test_flaml.py        # Tests FLAML
│   ├── test_h2o.py          # Tests H2O
│   ├── test_tpot.py         # Tests TPOT
│   └── test_feature_engineering.py
│
└── integration/             # Tests d'intégration
    ├── test_analyse.py      # Pipeline d'analyse complet
    ├── test_llmfe.py        # Feature Engineering via LLM
    ├── test_pipeline.py     # Pipeline AutoML complet
    └── test_automl.py       # Tests AutoML individuels
```

---

## Tests d'intégration

### Test Analyse

Génère l'analyse statistique complète d'un dataset.

#### Commandes

```bash
# Analyse basique (stats uniquement) - Titanic par défaut
python tests/integration/test_analyse.py

# Choisir un projet spécifique
python tests/integration/test_analyse.py --project titanic
python tests/integration/test_analyse.py --project verbatims
python tests/integration/test_analyse.py --project cate_metier
python tests/integration/test_analyse.py --project avis_client

# Avec analyse LLM interactive
python tests/integration/test_analyse.py --with-llm --provider openai --model gpt-4o-mini

# Analyser tous les projets
python tests/integration/test_analyse.py --all

# Mode silencieux
python tests/integration/test_analyse.py -q
```

#### Options disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--project`, `-p` | Projet à analyser | `titanic` |
| `--only-stats` | Stats uniquement (sans LLM) | `True` |
| `--with-llm` | Active l'analyse LLM interactive | `False` |
| `--provider` | Provider LLM (`openai`, `ollama`) | `openai` |
| `--model` | Modèle LLM | `gpt-4o-mini` |
| `--quiet`, `-q` | Mode silencieux | `False` |
| `--all` | Analyser tous les projets | `False` |

#### Outputs générés

```
outputs/analyse/{projet}/{timestamp}/
├── stats/
│   └── report_stats.json    # Rapport statistique complet
├── full/
│   └── report_full.json     # Rapport avec annotations LLM (si --with-llm)
├── logs/
│   └── analyse.log          # Log d'exécution
└── metadata.json            # Métadonnées de l'analyse
```

---

### Test LLMFE

Exécute le module LLMFE (LLM-based Feature Engineering) qui génère automatiquement des features via LLM et algorithme évolutif.

#### Principe de fonctionnement

LLMFE utilise un LLM (OpenAI) pour générer des fonctions de feature engineering, puis évalue ces fonctions avec XGBoost en cross-validation. Un algorithme évolutif sélectionne et améliore les meilleures fonctions.

**Architecture des données** : LLMFE utilise `src/analyse/` comme **source unique de vérité** pour les statistiques et corrélations des features. Si aucune analyse n'existe, elle est générée automatiquement.

#### Commandes

```bash
# Test basique avec dry-run (valide la config sans exécuter)
python tests/integration/test_llmfe.py --dry-run

# Exécution sur Titanic (par défaut) - format basic
python tests/integration/test_llmfe.py

# Choisir un projet
python tests/integration/test_llmfe.py --project titanic
python tests/integration/test_llmfe.py --project verbatims

# Configurer le nombre d'itérations LLM
python tests/integration/test_llmfe.py --max-samples 20

# Choisir le modèle OpenAI
python tests/integration/test_llmfe.py --model gpt-4o-mini
python tests/integration/test_llmfe.py --model gpt-4o
python tests/integration/test_llmfe.py --model gpt-3.5-turbo
```

#### Formats de prompt enrichis

LLMFE supporte 3 formats de présentation des features dans le prompt LLM :

| Format | Description | Usage |
|--------|-------------|-------|
| `basic` | Format simple (nom, type, range) | Par défaut, économique en tokens |
| `tags` | Format compact avec tags inline | Bon compromis info/tokens |
| `hierarchical` | Format structuré par importance | Maximum d'informations |

```bash
# Format basic (défaut)
python tests/integration/test_llmfe.py -n 5

# Format tags (compact avec indicateurs)
python tests/integration/test_llmfe.py -n 5 --feature-format tags

# Format hierarchical (structuré par importance)
python tests/integration/test_llmfe.py -n 5 --feature-format hierarchical
```

**Exemples de sortie par format :**

```
# BASIC
- Age: Age (numerical variable within range [0.42, 80.0])
- Sex: Sex (categorical variable with categories [male, female])

# TAGS
- Age: Age [NUM] [20% NaN] [CORR:-0.08] (0.42 to 80.00)
- Sex: Sex [CAT] [HIGH_PRED] [CORR:0.54] (male, female)

# HIERARCHICAL
### High-Value Features (strong predictors):
- Sex: Sex [CORR: 0.54] (male/female)

### Medium-Value Features:
- Age: Age [20% missing] (0.4-80.0)

### Low-Value Features (consider dropping):
- PassengerId: Passenger Id [ID - no predictive value]
```

#### Réutiliser une analyse existante

Pour de meilleurs résultats, lancez d'abord l'analyse puis réutilisez-la :

```bash
# 1. Lancer l'analyse (génère report_stats.json)
python tests/integration/test_analyse.py -p titanic

# 2. LLMFE avec l'analyse existante
python tests/integration/test_llmfe.py -n 10 --feature-format hierarchical \
    --analyse-path outputs/titanic/analyse/TIMESTAMP/stats/report_stats.json
```

Si `--analyse-path` n'est pas fourni et qu'un format enrichi est demandé, LLMFE lance automatiquement l'analyse via `src/analyse/`.

#### Options disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--project`, `-p` | Projet à traiter | `titanic` |
| `--max-samples`, `-n` | Nombre maximum d'itérations LLM | `10` |
| `--model`, `-m` | Modèle OpenAI à utiliser | `gpt-4o-mini` |
| `--samples-per-prompt` | Nombre de samples générés par appel API | `2` |
| `--feature-format`, `-f` | Format des features (`basic`, `tags`, `hierarchical`) | `basic` |
| `--analyse-path` | Chemin vers un JSON d'analyse existant | `None` |
| `--dry-run` | Valide la configuration sans exécuter | `False` |
| `--quiet`, `-q` | Mode silencieux | `False` |

#### Prérequis spécifiques

- **OPENAI_API_KEY** : Clé API OpenAI obligatoire
- Le module `llmfe` doit être présent dans `src/feature_engineering/llmfe/`
- Le module `analyse` doit être présent dans `src/analyse/` (utilisé automatiquement)

#### Outputs générés

```
outputs/{projet}/
├── analyse/{timestamp}/              # Généré automatiquement si format enrichi
│   ├── stats/
│   │   └── report_stats.json         # Rapport statistique (source des insights)
│   ├── full/
│   └── logs/
│
└── feature_engineering/{timestamp}/
    ├── features/
    │   ├── train_fe.parquet          # Train avec features générées
    │   └── test_fe.parquet           # Test avec features générées
    ├── llmfe/
    │   ├── samples/                  # Historique des fonctions générées
    │   │   ├── sample_0000.json
    │   │   ├── sample_0001.json
    │   │   └── ...
    │   ├── tensorboard/              # Logs TensorBoard
    │   └── results/
    │       ├── best_model.json       # Meilleure fonction trouvée
    │       ├── all_scores.json       # Tous les scores
    │       ├── evolution_history.json # Historique de l'évolution
    │       └── evolution_report.html  # Rapport HTML
    ├── transforms/
    │   └── pipeline.pkl              # Pipeline de transformation
    ├── specs/
    │   └── specification.txt         # Spec LLMFE générée
    ├── logs/
    │   └── feature_engineering.log
    └── metadata.json
```

#### Exemple de sortie

```
======================================================================
  LLMFE - TITANIC
  Dataset Titanic - Prédiction de survie
======================================================================

[INFO] Dossier de sortie: outputs/titanic/feature_engineering/20251130_213623

[1/5] Chargement des données depuis data/raw/titanic...
    - Lignes train: 891
    - Lignes test: 418
    - Features: 11
    - Cible: Survived

[2/5] Préparation des métadonnées...
    Features avec métadonnées: 11

[3/5] Configuration LLMFE...
    - Modèle: gpt-3.5-turbo
    - Max samples: 5
    - Samples/prompt: 2
    - Type: Classification

[4/5] Exécution de LLMFE...
📊 Lancement de l'analyse via src/analyse/...
✅ Analyse sauvegardée: outputs/titanic/analyse/20251130_213623/stats/report_stats.json
📈 Calcul des corrélations avancées...
✅ Corrélations calculées pour 11 features
📝 Format des features: tags

============================================================
           DÉMARRAGE DE LLMFE
============================================================

====================================================================================================
                           ÉVOLUTION DES FEATURES LLMFE
====================================================================================================

#    Score        Delta      Features Créées                Features Supprimées
----------------------------------------------------------------------------------------------------
-    0.8148       🏆 +0.8148    -                              -
2    0.8193       🏆 +0.0044    Family_Size, Is_Alone, Title   -
----------------------------------------------------------------------------------------------------

📊 Résumé:
   • Meilleur score: 0.8193 (sample #2)
   • Amélioration: +0.0044
   • Taux de succès: 100.0%
====================================================================================================

[5/5] Sauvegarde des métadonnées...

======================================================================
  LLMFE TERMINE
======================================================================
```

---

### Test Pipeline Complet (Full Pipeline)

Exécute le **pipeline complet** : Analyse -> Feature Engineering -> AutoML en une seule commande.

**Principe clé** : Les paramètres (`task_type`, `metric`, `feature_format`, `max_samples`, `time_budget`) sont **auto-détectés** depuis l'analyse du dataset. Seuls `project_name` et `target_col` sont obligatoires.

#### Commandes

```bash
# Test analyse seule (rapide)
python tests/integration/test_pipeline_all.py --analyse-only

# Test analyse + Feature Engineering (sans AutoML)
python tests/integration/test_pipeline_all.py --no-automl

# Test complet (analyse + FE + AutoML)
python tests/integration/test_pipeline_all.py --full

# Tests unitaires de DetectedParams
python tests/integration/test_pipeline_all.py --unit-tests
```

#### Options disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--analyse-only` | Test de l'analyse seule | `False` |
| `--no-automl` | Analyse + FE (sans AutoML) | `False` |
| `--full` | Pipeline complet | `False` |
| `--unit-tests` | Tests unitaires DetectedParams | `False` |
| `--override-metric` | Force une métrique | `None` |
| `--override-max-samples` | Force le nb d'itérations LLMFE | `3` |
| `--override-time-budget` | Force le time budget AutoML (sec) | `30` |

#### Auto-détection des paramètres

Le pipeline analyse le dataset et détecte automatiquement :

| Paramètre | Source | Logique |
|-----------|--------|---------|
| `task_type` | `target.problem_type` | "regression" si contient "regression", sinon "classification" |
| `metric` | `target.is_imbalanced` + `n_classes` | accuracy (équilibré), f1 (déséquilibré binaire), f1_macro (déséquilibré multiclasse), rmse (regression) |
| `feature_format` | Complexité du dataset | basic (<=5 features), tags (moyen), hierarchical (complexe) |
| `max_samples` | `n_features` | 10 (<=10), 15 (moyen), 25 (>30) |
| `time_budget` | `n_rows` | 60s (<1000), 120s (moyen), 300s (>50000) |

#### Personnaliser les seuils d'inférence

Vous pouvez ajuster les seuils dans `src/pipeline/pipeline_all.py` (classe `InferenceConfig`, lignes 29-88) :

```python
@dataclass
class InferenceConfig:
    # FEATURE FORMAT
    format_basic_max_features: int = 5          # basic si <= 5 features
    format_hierarchical_threshold: float = 0.5  # hierarchical si complexité > 0.5

    # MAX SAMPLES (itérations LLMFE)
    max_samples_few_features_threshold: int = 10
    max_samples_many_features_threshold: int = 30
    max_samples_small: int = 10
    max_samples_medium: int = 15
    max_samples_large: int = 25

    # TIME BUDGET (AutoML en secondes)
    time_budget_small_rows_threshold: int = 1000
    time_budget_large_rows_threshold: int = 50000
    time_budget_small: int = 60
    time_budget_medium: int = 120
    time_budget_large: int = 300
```

#### Usage programmatique

```python
from src.pipeline.pipeline_all import run_pipeline, InferenceConfig

# Minimal - tout auto-détecté
result = run_pipeline(
    project_name="titanic",
    df_train=df,
    target_col="Survived",
)

# Avec overrides directs
result = run_pipeline(
    "titanic", df, "Survived",
    override_max_samples=5,        # Force 5 itérations LLMFE
    override_time_budget=30,       # Force 30s pour AutoML
    override_feature_format="tags", # Force format tags
)

# Avec config d'inférence personnalisée
config = InferenceConfig(
    max_samples_small=5,
    time_budget_large=600,
)
result = run_pipeline("titanic", df, "Survived", inference_config=config)

# Accéder aux résultats
print(f"Task détecté: {result.detected_params.task_type}")
print(f"Metric utilisée: {result.detected_params.metric}")
print(f"Best framework: {result.best_framework}")
print(f"Best score: {result.best_score}")
```

#### Outputs générés

```
outputs/{projet}/{timestamp}/
├── {projet}/
│   └── analyse/{timestamp}/
│       └── stats/
│           └── report_stats.json    # Rapport d'analyse (source auto-détection)
├── feature_engineering/{timestamp}/
│   ├── llmfe/
│   │   └── results/
│   │       └── best_model.json
│   └── features/
│       └── train_fe.parquet
├── models/                          # Modèles AutoML
│   ├── flaml/
│   └── autogluon/
└── pipeline_summary.json            # Résumé complet du pipeline
```

#### Exemple de sortie

```
######################################################################
#  FULL PIPELINE - TITANIC
######################################################################
#
#  Timestamp: 20251201_012557
#  Output:    outputs/titanic/20251201_012557
#  Target:    Survived
#  Steps:     Analyse -> FE(llmfe) -> AutoML(flaml,autogluon)
#
#  Note: task_type, metric, feature_format seront AUTO-DÉTECTÉS
#
######################################################################

============================================================
  ÉTAPE 1: ANALYSE DU DATASET
============================================================
  Target: Survived
  Lignes: 891
  Colonnes: 12
  ...
  Rapport sauvegardé: outputs/titanic/analyse/20251201_012557/stats/report_stats.json

----------------------------------------
  Chargement des paramètres détectés...

  Paramètres détectés depuis l'analyse:
  ─────────────────────────────────────
  Target:         Survived
  Problem type:   binary_classification
  Task type:      classification
  Metric:         accuracy
  Imbalanced:     False (ratio: 1.61)

  Dataset:        891 rows, 11 features
  Feature format: tags
  Max samples:    15
  Time budget:    60s

============================================================
  ÉTAPE 2: FEATURE ENGINEERING (LLMFE)
============================================================
  Task type:      classification (auto-détecté)
  Feature format: tags (auto-détecté)
  Max samples:    15 (auto-détecté)
  ...

============================================================
  ÉTAPE 3: AUTOML
============================================================
  Metric:       accuracy (auto-détecté)
  Time budget:  60s (auto-détecté)
  Frameworks:   flaml, autogluon
  ...

######################################################################
#  PIPELINE TERMINÉ
######################################################################
#
#  Résultats dans: outputs/titanic/20251201_012557
#  Task type:      classification
#  Metric:         accuracy
#  Best framework: autogluon
#  Best score:     0.8352
#
######################################################################
```

---

### Test Pipeline AutoML (Legacy)

Exécute le pipeline AutoML seul (sans analyse ni FE).

#### Commande

```bash
python tests/integration/test_pipeline.py
```

#### Frameworks exécutés

1. **FLAML** - Fast AutoML
2. **AutoGluon** - Amazon AutoML
3. **TPOT** - Genetic programming AutoML
4. **H2O** - Distributed AutoML

#### Outputs générés

```
Modeles/{projet}/
├── flaml/
│   └── time_budget_{N}/
├── autogluon/
│   └── time_budget_{N}/
├── tpot/
│   └── time_budget_{N}/
├── h2o/
│   └── time_budget_{N}/
└── logs/
    └── automl_{date}.log
```

---

### Test AutoML individuel

Teste un framework AutoML spécifique.

#### Commandes

```bash
# Via pytest
pytest tests/integration/test_automl.py -v

# Ou exécution directe
python tests/integration/test_automl.py
```

---

## Tests unitaires

### Exécuter tous les tests unitaires

```bash
pytest tests/unit/ -v
```

### Exécuter un test spécifique

```bash
# Test du module analyse
pytest tests/unit/test_analyse.py -v

# Test AutoGluon uniquement
pytest tests/unit/test_autogluon.py -v

# Test FLAML uniquement
pytest tests/unit/test_flaml.py -v

# Test H2O uniquement
pytest tests/unit/test_h2o.py -v

# Test TPOT uniquement
pytest tests/unit/test_tpot.py -v
```

### Tests avec marqueurs

```bash
# Ignorer les tests qui nécessitent une clé API
pytest tests/unit/ -v -m "not requires_api"

# Tests rapides uniquement
pytest tests/unit/ -v -m "fast"
```

---

## Outputs générés

### Structure globale

```
project_root/
├── outputs/
│   ├── {projet}/
│   │   ├── analyse/              # Résultats d'analyse
│   │   │   └── {timestamp}/
│   │   ├── feature_engineering/  # Features générées (LLMFE)
│   │   │   └── {timestamp}/
│   │   └── automl/               # Modèles AutoML
│   │       └── {timestamp}/
│
└── Modeles/                      # (Legacy) Modèles entraînés
    └── {projet}/
        ├── flaml/
        ├── autogluon/
        ├── tpot/
        ├── h2o/
        └── logs/
```

### Fichiers JSON d'analyse

Le `report_stats.json` contient :

```json
{
  "context": {
    "name": "titanic",
    "business_description": "Dataset Titanic - Prédiction de survie"
  },
  "basic_stats": {
    "n_rows": 891,
    "n_features": 11,
    "missing_cell_ratio": 0.09
  },
  "target": {
    "name": "Survived",
    "problem_type": "binary_classification",
    "class_balance": {"0": 0.62, "1": 0.38}
  },
  "features": [
    {
      "name": "Age",
      "inferred_type": "numeric",
      "missing_rate": 0.19,
      "fe_hints": ["numeric_imputation", "candidate_for_scaling"]
    }
  ]
}
```

---

## Troubleshooting

### Erreur : Module not found

```bash
# S'assurer que le PYTHONPATH est configuré
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou exécuter depuis la racine du projet
cd /path/to/IA_create_IA_cate
python tests/integration/test_analyse.py
```

### Erreur : OPENAI_API_KEY non trouvée

```bash
# Vérifier que le fichier .env existe
cat .env

# Ou définir temporairement
export OPENAI_API_KEY="sk-..."
```

### Erreur : H2O BrokenPipeError

H2O peut avoir des problèmes de connexion. Solutions :

```bash
# Redémarrer le serveur H2O
python -c "import h2o; h2o.init()"

# Ou augmenter le timeout
python tests/integration/test_pipeline.py --time-budget 300
```

### Erreur : AutoGluon "label column not found"

Le dataset de test n'a pas de colonne cible (cas Kaggle). Le code gère ce cas automatiquement en utilisant le score de validation.

### Tests trop lents

```bash
# Réduire le time_budget
python tests/integration/test_pipeline.py --time-budget 30

# Ou tester un seul framework
pytest tests/unit/test_flaml.py -v
```

### Erreur : LLMFE "Impossible d'importer llmfe"

Vérifier que le module est présent :

```bash
ls src/feature_engineering/llmfe/

# Si le dossier n'existe pas, vérifier la structure
ls src/feature_engineering/
```

### Erreur : LLMFE Rate Limit OpenAI

Réduire le nombre de samples par prompt :

```bash
python tests/integration/test_llmfe.py --samples-per-prompt 1 --max-samples 5
```

### LLMFE : Tester sans consommer d'API

Utiliser le mode dry-run pour valider la configuration :

```bash
python tests/integration/test_llmfe.py --dry-run
```

### LLMFE : Erreur "Module analyse non trouvé"

Si vous utilisez `--feature-format tags` ou `hierarchical`, le module `src/analyse/` est requis :

```bash
# Vérifier que le module existe
ls src/analyse/

# Structure attendue
src/analyse/
├── path_config.py
├── statistiques/
│   └── report.py
└── correlation/
    └── correlation.py
```

### LLMFE : Réutiliser une analyse existante

Pour éviter de relancer l'analyse à chaque fois :

```bash
# 1. Lister les analyses existantes
ls outputs/titanic/analyse/

# 2. Utiliser le chemin du JSON
python tests/integration/test_llmfe.py -n 10 -f hierarchical \
    --analyse-path outputs/titanic/analyse/20251130_171231/stats/report_stats.json
```

---

## Commandes rapides

```bash
# Tout tester
pytest tests/ -v

# Tests unitaires uniquement
pytest tests/unit/ -v

# Tests d'intégration uniquement
pytest tests/integration/ -v

# Analyse rapide Titanic
python tests/integration/test_analyse.py -p titanic

# LLMFE avec dry-run (validation sans API)
python tests/integration/test_llmfe.py --dry-run

# LLMFE exécution complète (format basic)
python tests/integration/test_llmfe.py -p titanic -n 10

# LLMFE avec format enrichi (tags)
python tests/integration/test_llmfe.py -n 10 --feature-format tags

# LLMFE avec format hiérarchique (maximum d'info)
python tests/integration/test_llmfe.py -n 10 --feature-format hierarchical

# LLMFE avec analyse existante
python tests/integration/test_llmfe.py -n 10 -f hierarchical \
    --analyse-path outputs/titanic/analyse/TIMESTAMP/stats/report_stats.json

# =============================================
# PIPELINE COMPLET (recommandé)
# =============================================

# Pipeline complet - analyse seule (test rapide)
python tests/integration/test_pipeline_all.py --analyse-only

# Pipeline complet - analyse + FE (sans AutoML)
python tests/integration/test_pipeline_all.py --no-automl

# Pipeline complet - tout (analyse + FE + AutoML)
python tests/integration/test_pipeline_all.py --full

# Pipeline complet avec overrides
python tests/integration/test_pipeline_all.py --full \
    --override-max-samples 5 \
    --override-time-budget 30

# =============================================
# LEGACY
# =============================================

# Pipeline AutoML seul (sans analyse ni FE)
python tests/integration/test_pipeline.py
```

---

## Voir aussi

- [Architecture du projet](./architecture.md)
- [Guide de migration des imports](./import_migration.md)
- [README principal](../README.md)
