# IA Create IA

Pipeline ML automatisé : **Analyse → Feature Engineering → AutoML**

Un système intelligent qui transforme automatiquement vos données en modèles de machine learning performants, en utilisant des LLM pour orchestrer le processus.

---

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE COMPLET                            │
│                                                                     │
│   📊 Analyse    →    🔧 Feature Engineering    →    🤖 AutoML      │
│                                                                     │
│   Statistiques       LLMFE (génération         4 frameworks        │
│   + Détection        automatique de            en parallèle        │
│   automatique        features par LLM)                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prérequis

- Python 3.12+
- Conda (recommandé)

### Configuration

```bash
# Cloner le projet
git clone <repo-url>
cd IA_create_IA_cate

# Créer l'environnement conda
conda env create -f environment.yml
conda activate Ia_create_ia

# Configurer les clés API (optionnel, pour les features LLM)
cp .env.example .env
# Éditer .env avec votre clé OPENAI_API_KEY
```

---

## Démarrage rapide

### 1. Analyse simple

```bash
conda run -n Ia_create_ia python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only
```

### 2. Pipeline complet

```bash
conda run -n Ia_create_ia python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --full
```

### 3. Interface web

```bash
conda run -n Ia_create_ia streamlit run src/front/pipeline_streamlit.py
```

---

## Structure du projet

```
IA_create_IA_cate/
│
├── src/                    # Code source
│   ├── analyse/            # Module d'analyse statistique
│   ├── feature_engineering/# Module de génération de features
│   ├── automl/             # Module AutoML (4 frameworks)
│   ├── pipeline/           # Orchestration du pipeline
│   ├── core/               # Utilitaires partagés
│   ├── front/              # Interface Streamlit
│   └── few_shot/           # Module few-shot learning
│
├── data/raw/               # Datasets (titanic/, verbatims/, etc.)
├── outputs/                # Résultats générés
├── tests/                  # Tests unitaires et intégration
└── docs/                   # Documentation détaillée
```

---

## Les 5 modules principaux

### 1. Module Analyse (`src/analyse/`)

Analyse automatique de vos données pour comprendre leur structure.

**Ce qu'il fait :**
- Calcule les statistiques de base (lignes, colonnes, types)
- Détecte automatiquement le type de problème (classification/régression)
- Identifie les déséquilibres de classes
- Analyse les valeurs manquantes
- Calcule les corrélations (optionnel)
- Enrichit l'analyse avec un LLM (optionnel)

**Résultat :** Un rapport JSON qui guide les étapes suivantes.

---

### 2. Module Feature Engineering (`src/feature_engineering/`)

Génère automatiquement de nouvelles features pour améliorer les performances.

**Ce qu'il fait :**
- **LLMFE** : Utilise un LLM pour proposer de nouvelles features
- Transformations classiques (encodage, scaling, dates)
- Évalue chaque feature générée
- Garde les meilleures

**Formats de génération :**
| Format | Quand l'utiliser |
|--------|------------------|
| `basic` | Peu de features (≤5) |
| `tags` | Complexité moyenne (6-50 features) |
| `hierarchical` | Beaucoup de features ou données textuelles |

---

### 3. Module AutoML (`src/automl/`)

Entraîne automatiquement des modèles avec 4 frameworks différents.

**Frameworks supportés :**

| Framework | Description |
|-----------|-------------|
| **FLAML** | Rapide, léger, idéal pour commencer |
| **AutoGluon** | Performant, stacking automatique |
| **TPOT** | Optimisation génétique de pipelines |
| **H2O** | Robuste, export MOJO pour production |

**Ce qu'il fait :**
- Lance chaque framework en parallèle
- Compare les résultats
- Sauvegarde le meilleur modèle

---

### 4. Module Pipeline (`src/pipeline/`)

Orchestre les 3 modules précédents de manière intelligente.

**Ce qu'il fait :**
- Lit le rapport d'analyse
- Détecte automatiquement les meilleurs paramètres
- Enchaîne les étapes dans le bon ordre
- Gère les erreurs gracieusement

**Paramètres auto-détectés :**
- Type de tâche (classification/régression)
- Métrique optimale (accuracy, f1, rmse...)
- Format de features
- Budget temps

---

### 5. Module Frontend (`src/front/`)

Interface web pour utiliser le pipeline sans code.

**Fonctionnalités :**
- Upload de fichiers CSV
- Sélection de la colonne cible
- Suivi du progrès en temps réel
- Visualisation des résultats

**Lancement :**
```bash
conda run -n Ia_create_ia streamlit run src/front/pipeline_streamlit.py
```

---

## Référence CLI

Le pipeline se contrôle via la ligne de commande avec de nombreuses options.

### Commandes de base

```bash
# Remplacer [OPTIONS] par les options ci-dessous
conda run -n Ia_create_ia python tests/integration/test_pipeline_all.py [OPTIONS]
```

---

### Paramètres obligatoires

| Option | Description |
|--------|-------------|
| `--dataset NOM` | Nom du dossier dans `data/raw/` |
| `--target COLONNE` | Nom de la colonne à prédire |

**Exemple :**
```bash
--dataset titanic --target Survived
```

---

### Modes d'exécution

| Option | Description |
|--------|-------------|
| `--analyse-only` | Analyse statistique uniquement |
| `--no-automl` | Analyse + Feature Engineering (sans AutoML) |
| `--no-fe` | Analyse + AutoML (sans Feature Engineering) |
| `--full` | Pipeline complet (Analyse + FE + AutoML) |
| `--force-analyse` | Regénère l'analyse même si elle existe |

**Exemples :**
```bash
# Analyse seule
--dataset titanic --target Survived --analyse-only

# Pipeline complet
--dataset titanic --target Survived --full
```

---

### Configuration du projet

| Option | Description | Défaut |
|--------|-------------|--------|
| `--project NOM` | Nom du projet | Nom du dataset |
| `--output-dir CHEMIN` | Dossier de sortie | `outputs` |

---

### Configuration de l'analyse

| Option | Description |
|--------|-------------|
| `--with-correlations` | Active le calcul des corrélations |
| `--correlation-methods LISTE` | Méthodes : `pearson,spearman,kendall,mutual_info,mic,phik` |

**Exemple :**
```bash
--analyse-only --with-correlations --correlation-methods pearson,spearman
```

---

### Configuration LLM

| Option | Description | Défaut |
|--------|-------------|--------|
| `--with-llm` | Active l'analyse métier par LLM | Désactivé |
| `--analyse-provider` | Provider LLM | `openai` |
| `--analyse-model` | Modèle pour l'analyse | `gpt-4o-mini` |
| `--llmfe-model` | Modèle pour le Feature Engineering | `gpt-3.5-turbo` |

**Exemple :**
```bash
--with-llm --analyse-model gpt-4o
```

---

### Overrides (forcer des valeurs)

Ces options permettent de forcer des paramètres au lieu de les laisser être auto-détectés.

| Option | Description | Valeurs possibles |
|--------|-------------|-------------------|
| `--override-task-type` | Force le type de tâche | `classification`, `regression` |
| `--override-metric` | Force la métrique | `f1`, `accuracy`, `rmse`, `roc_auc`... |
| `--override-feature-format` | Force le format | `basic`, `tags`, `hierarchical` |
| `--override-max-samples` | Itérations LLMFE | Nombre entier (défaut: 3) |
| `--override-time-budget` | Budget temps AutoML | Secondes (défaut: 60) |

**Exemple :**
```bash
--full --override-metric f1 --override-time-budget 120
```

---

### Configuration AutoML

#### Général

| Option | Description | Défaut |
|--------|-------------|--------|
| `--automl-frameworks` | Frameworks à utiliser | `flaml,autogluon` |

**Frameworks disponibles :** `flaml`, `autogluon`, `tpot`, `h2o`

**Exemple :**
```bash
--automl-frameworks flaml,autogluon,h2o
```

#### FLAML

| Option | Description |
|--------|-------------|
| `--flaml-time-budget` | Budget temps en secondes |
| `--flaml-metric` | Métrique d'optimisation |

#### AutoGluon

| Option | Description | Défaut |
|--------|-------------|--------|
| `--autogluon-presets` | Preset de qualité | `medium_quality_faster_train` |
| `--autogluon-time-budget` | Budget temps en secondes | - |

**Presets disponibles :** `best_quality`, `high_quality`, `good_quality`, `medium_quality_faster_train`

#### TPOT

| Option | Description | Défaut |
|--------|-------------|--------|
| `--tpot-generations` | Nombre de générations | 7 |
| `--tpot-population-size` | Taille de population | 25 |
| `--tpot-cv` | Folds de cross-validation | 5 |

#### H2O

| Option | Description | Défaut |
|--------|-------------|--------|
| `--h2o-time-budget` | Budget temps en secondes | - |
| `--h2o-verbosity` | Niveau de log | `info` |
| `--h2o-no-mojo` | Désactive l'export MOJO | - |

---

### Seuils d'analyse (avancé)

| Option | Description | Défaut |
|--------|-------------|--------|
| `--high-cardinality-threshold` | Seuil haute cardinalité | 50 |
| `--high-missing-threshold` | Seuil valeurs manquantes | 0.3 |
| `--strong-corr-threshold` | Seuil corrélation forte | 0.8 |

---

## Exemples complets

### Analyse rapide d'un dataset

```bash
conda run -n Ia_create_ia python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only
```

### Analyse avec corrélations et LLM

```bash
conda run -n Ia_create_ia python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only \
    --with-correlations \
    --with-llm
```

### Pipeline complet avec configuration personnalisée

```bash
conda run -n Ia_create_ia python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --full \
    --override-metric f1 \
    --override-time-budget 120 \
    --automl-frameworks flaml,autogluon,h2o
```

### Feature Engineering seul (sans AutoML)

```bash
conda run -n Ia_create_ia python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --no-automl \
    --override-max-samples 10
```

---

## Structure des résultats

Après exécution, les résultats sont organisés ainsi :

```
outputs/{project_name}/
│
├── analyse/
│   ├── stats/
│   │   └── report_stats.json      # Rapport statistique
│   ├── full/
│   │   └── report_full.json       # Rapport enrichi LLM
│   └── agent_llm/
│       └── conversation.json      # Historique LLM
│
├── feature_engineering/
│   ├── features/
│   │   ├── train_fe.parquet       # Features train
│   │   └── test_fe.parquet        # Features test
│   └── llmfe/
│       └── results/               # Résultats LLMFE
│
├── automl/
│   ├── flaml/
│   ├── autogluon/
│   ├── tpot/
│   ├── h2o/
│   └── results/
│       └── comparison.json        # Comparaison des frameworks
│
└── models/                        # Modèles sauvegardés
```

---

## Documentation supplémentaire

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli_reference.md) | Référence complète CLI |
| [Business Agent](docs/business_agent.md) | Agent de clarification métier |
| [Testing](docs/testing.md) | Guide des tests |

---

## Environnement

```bash
# Activer l'environnement
conda activate Ia_create_ia

# Vérifier l'installation
conda run -n Ia_create_ia python -c "import pandas; print('OK')"
```

---

## Licence

Projet personnel - Tous droits réservés.
