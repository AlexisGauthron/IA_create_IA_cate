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

- Python 3.9
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
conda activate Ia_create_ia
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only
```

### 2. Pipeline complet

```bash
conda activate Ia_create_ia
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --full
```

### 3. Interface web

```bash
conda activate Ia_create_ia
streamlit run src/front/pipeline_streamlit.py
```

---

## Structure du projet

```
IA_create_IA_cate/
│
├── src/                    # Code source
│   ├── analyse/            # Module d'analyse statistique
│   ├── feature_engineering/# Module de génération de features (LLMFE, DFS, Hybrid)
│   ├── automl/             # Module AutoML (4 frameworks)
│   ├── models/             # Module modèles partagé (évaluation multi-modèle)
│   ├── pipeline/           # Orchestration du pipeline
│   ├── core/               # Utilitaires partagés
│   └── front/              # Interface Streamlit
│
├── data/raw/               # Datasets (titanic/, verbatims/, etc.)
├── outputs/                # Résultats générés
├── tests/                  # Tests unitaires et intégration
└── docs/                   # Documentation détaillée
```

---

## Les modules principaux

### 1. Module Analyse (`src/analyse/`)

Analyse automatique de vos données pour comprendre leur structure.

- Calcule les statistiques de base (lignes, colonnes, types)
- Détecte automatiquement le type de problème (classification/régression)
- Identifie les déséquilibres de classes
- Analyse les valeurs manquantes
- Calcule les corrélations (optionnel)
- Enrichit l'analyse avec un LLM (optionnel)

---

### 2. Module Feature Engineering (`src/feature_engineering/`)

Génère automatiquement de nouvelles features pour améliorer les performances.

**Approches disponibles :**
- **LLMFE** : Utilise un LLM pour proposer de nouvelles features (approche évolutionnaire)
- **DFS** : Deep Feature Synthesis via FeatureTools
- **Hybrid** : Combinaison LLMFE + DFS

| Format | Quand l'utiliser |
|--------|------------------|
| `basic` | Peu de features (≤5) |
| `tags` | Complexité moyenne (6-50 features) |
| `hierarchical` | Beaucoup de features ou données textuelles |

---

### 3. Module AutoML (`src/automl/`)

Entraîne automatiquement des modèles avec 4 frameworks différents.

| Framework | Description |
|-----------|-------------|
| **FLAML** | Rapide, léger, idéal pour commencer |
| **AutoGluon** | Performant, stacking automatique |
| **TPOT** | Optimisation génétique de pipelines |
| **H2O** | Robuste, export MOJO pour production |

---

### 4. Module Models (`src/models/`)

Module centralisé pour l'évaluation multi-modèle.

- Wrappers uniformes (XGBoost, LightGBM, CatBoost, sklearn)
- Cross-validation multi-métrique pondérée
- Support classification et régression

---

### 5. Module Frontend (`src/front/`)

Interface web Streamlit pour utiliser le pipeline sans code.

```bash
conda activate Ia_create_ia
streamlit run src/front/pipeline_streamlit.py
```

---

## Référence CLI

### Paramètres obligatoires

| Option | Description |
|--------|-------------|
| `--dataset NOM` | Nom du dossier dans `data/raw/` |
| `--target COLONNE` | Nom de la colonne à prédire |

### Modes d'exécution

| Option | Description |
|--------|-------------|
| `--analyse-only` | Analyse statistique uniquement |
| `--no-automl` | Analyse + Feature Engineering (sans AutoML) |
| `--no-fe` | Analyse + AutoML (sans Feature Engineering) |
| `--full` | Pipeline complet (Analyse + FE + AutoML) |

### Configuration LLM

| Option | Description | Défaut |
|--------|-------------|--------|
| `--with-llm` | Active l'analyse métier par LLM | Désactivé |
| `--llmfe-model` | Modèle pour le Feature Engineering | `gpt-3.5-turbo` |

### Overrides

| Option | Description |
|--------|-------------|
| `--override-metric` | Force la métrique (`f1`, `accuracy`, `rmse`...) |
| `--override-max-samples` | Itérations LLMFE (défaut: 3) |
| `--override-time-budget` | Budget temps AutoML en secondes (défaut: 60) |

📚 **Référence complète** : [docs/cli_reference.md](docs/cli_reference.md)

---

## Exemples

```bash
# Analyse rapide
python tests/integration/test_pipeline_all.py \
    --dataset titanic --target Survived --analyse-only

# Avec corrélations et LLM
python tests/integration/test_pipeline_all.py \
    --dataset titanic --target Survived --analyse-only \
    --with-correlations --with-llm

# Pipeline complet
python tests/integration/test_pipeline_all.py \
    --dataset titanic --target Survived --full \
    --override-metric f1 --override-time-budget 120
```

---

## Structure des résultats

```
outputs/{project_name}/
├── analyse/
│   └── stats/report_stats.json      # Rapport statistique
├── feature_engineering/
│   ├── features/                    # Features générées (parquet)
│   └── llmfe/results/               # Résultats LLMFE
├── automl/
│   ├── flaml/
│   ├── autogluon/
│   └── ...
└── pipeline_summary.json            # Résumé du pipeline
```

---

## Sources et références

Ce projet utilise et s'inspire de plusieurs travaux et librairies open-source.

### Feature Engineering

#### LLM-FE (LLMFE)
Le module de Feature Engineering par LLM est basé sur le papier :

> **LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers**
> Nikhil Abhyankar, Parshin Shojaee, Chandan K. Reddy
> arXiv:2503.14434, 2025
>
> - Paper: [https://arxiv.org/abs/2503.14434](https://arxiv.org/abs/2503.14434)
> - HuggingFace: [https://huggingface.co/papers/2503.14434](https://huggingface.co/papers/2503.14434)

LLM-FE est lui-même construit sur :
- **FunSearch** (Google DeepMind): [github.com/google-deepmind/funsearch](https://github.com/google-deepmind/funsearch)
- **LLM-SR**: [github.com/deep-symbolic-mathematics/llm-sr](https://github.com/deep-symbolic-mathematics/llm-sr)

#### FeatureTools (DFS)
> **Deep Feature Synthesis**
> Feature Labs / Alteryx
> - Documentation: [https://featuretools.alteryx.com/](https://featuretools.alteryx.com/)
> - GitHub: [github.com/alteryx/featuretools](https://github.com/alteryx/featuretools)

---

### Frameworks AutoML

#### FLAML
> **FLAML: A Fast and Lightweight AutoML Library**
> Microsoft Research
> Chi Wang, Qingyun Wu, et al.
>
> - Paper: [https://arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095)
> - GitHub: [github.com/microsoft/FLAML](https://github.com/microsoft/FLAML)
> - Documentation: [https://microsoft.github.io/FLAML/](https://microsoft.github.io/FLAML/)

#### AutoGluon
> **AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data**
> Amazon Web Services
> Nick Erickson, et al.
>
> - Paper: [https://arxiv.org/abs/2003.06505](https://arxiv.org/abs/2003.06505)
> - GitHub: [github.com/autogluon/autogluon](https://github.com/autogluon/autogluon)
> - Documentation: [https://auto.gluon.ai/](https://auto.gluon.ai/)

#### TPOT
> **TPOT: A Tree-based Pipeline Optimization Tool for Automating Machine Learning**
> Randal S. Olson, et al.
>
> - Paper: [https://doi.org/10.1007/978-3-319-31204-0_9](https://doi.org/10.1007/978-3-319-31204-0_9)
> - GitHub: [github.com/EpistasisLab/tpot](https://github.com/EpistasisLab/tpot)
> - Documentation: [http://epistasislab.github.io/tpot/](http://epistasislab.github.io/tpot/)

#### H2O AutoML
> **H2O AutoML: Scalable Automatic Machine Learning**
> H2O.ai
>
> - GitHub: [github.com/h2oai/h2o-3](https://github.com/h2oai/h2o-3)
> - Documentation: [https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

---

### Modèles ML

#### XGBoost
> **XGBoost: A Scalable Tree Boosting System**
> Tianqi Chen, Carlos Guestrin
>
> - Paper: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
> - GitHub: [github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)

#### LightGBM
> **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**
> Microsoft
> Guolin Ke, et al.
>
> - Paper: [https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
> - GitHub: [github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)

#### CatBoost
> **CatBoost: unbiased boosting with categorical features**
> Yandex
> Liudmila Prokhorenkova, et al.
>
> - Paper: [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)
> - GitHub: [github.com/catboost/catboost](https://github.com/catboost/catboost)

#### Scikit-learn
> **Scikit-learn: Machine Learning in Python**
> Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
>
> - GitHub: [github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
> - Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)

---

### LLM Providers

#### OpenAI API
> - Documentation: [https://platform.openai.com/docs/](https://platform.openai.com/docs/)
> - Modèles utilisés: `gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4o`

#### Ollama (Local LLM)
> - GitHub: [github.com/ollama/ollama](https://github.com/ollama/ollama)
> - Documentation: [https://ollama.ai/](https://ollama.ai/)

---

### Interface & Visualisation

#### Streamlit
> - GitHub: [github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)
> - Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)

#### TensorBoard
> - Documentation: [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)
> - Utilisé pour le suivi des métriques LLMFE

---

### Librairies Data Science

| Librairie | Usage | Lien |
|-----------|-------|------|
| **Pandas** | Manipulation de données | [pandas.pydata.org](https://pandas.pydata.org/) |
| **NumPy** | Calculs numériques | [numpy.org](https://numpy.org/) |
| **Polars** | DataFrames performants | [pola.rs](https://pola.rs/) |
| **PyArrow** | Format Parquet | [arrow.apache.org](https://arrow.apache.org/) |
| **Matplotlib** | Visualisation | [matplotlib.org](https://matplotlib.org/) |
| **Seaborn** | Visualisation statistique | [seaborn.pydata.org](https://seaborn.pydata.org/) |

---

## Documentation supplémentaire

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli_reference.md) | Référence complète CLI |
| [Testing](docs/TESTING.md) | Guide des tests |
| [Architecture](docs/architecture.md) | Architecture du projet |

---

## Licence

Projet personnel - Tous droits réservés.

Les dépendances et sources citées sont sous leurs licences respectives (MIT, Apache 2.0, BSD, etc.).
