# Architecture Technique - Vue d'Ensemble

> Ce document est destiné aux **développeurs** qui veulent comprendre comment le projet est structuré.
> Pour l'utilisation (CLI, Streamlit), voir le [README principal](../../README.md).

---

## Vision du Projet

**IA Create IA** est un pipeline AutoML intelligent qui transforme un dataset brut en modèle ML performant, en utilisant des LLM pour :
- Comprendre le contexte métier
- Générer automatiquement des features pertinentes
- Recommander les métriques d'évaluation

---

## Flux Global du Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRÉE                                          │
│                    DataFrame + colonne cible                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 : ANALYSE                                              src/analyse/ │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Profilage statistique (types, distributions, valeurs manquantes)         │
│  • Détection du type de problème (classification/régression)                │
│  • Détection du déséquilibre de classes                                     │
│  • Corrélations (optionnel)                                                 │
│  • Enrichissement LLM (optionnel) : contexte métier, métrique recommandée   │
│                                                                             │
│  Output: outputs/{projet}/analyse/stats/report_stats.json                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 : FEATURE ENGINEERING                       src/feature_engineering/ │
├─────────────────────────────────────────────────────────────────────────────┤
│  • LLMFE : Un LLM génère du code Python pour créer des features             │
│  • Évaluation multi-modèle (XGBoost, LightGBM, RandomForest...)             │
│  • Sélection des meilleures features                                        │
│  • Support multi-métrique pondérée (ex: 60% recall + 40% precision)         │
│                                                                             │
│  Output: outputs/{projet}/feature_engineering/features/train_fe.parquet     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3 : AUTOML                                               src/automl/ │
├─────────────────────────────────────────────────────────────────────────────┤
│  • 4 frameworks en parallèle : FLAML, AutoGluon, TPOT, H2O                  │
│  • Comparaison automatique des performances                                 │
│  • Export du meilleur modèle (+ MOJO pour H2O)                              │
│                                                                             │
│  Output: outputs/{projet}/automl/{framework}/model.*                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SORTIE                                          │
│              Modèle entraîné + Rapport de comparaison                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Les 7 Modules et Leur Rôle

| Module | Responsabilité | Dépend de |
|--------|----------------|-----------|
| **`core/`** | Fondations : config, LLM client, I/O, chemins | Aucun |
| **`models/`** | Interface unifiée pour modèles ML + évaluation | `core/` |
| **`analyse/`** | Profilage statistique + enrichissement LLM | `core/` |
| **`feature_engineering/`** | Génération de features (LLMFE, DFS, transforms) | `core/`, `models/` |
| **`automl/`** | Orchestration des 4 frameworks AutoML | `core/` |
| **`pipeline/`** | Orchestration globale (Analyse → FE → AutoML) | Tous |
| **`front/`** | Interface Streamlit | Tous |

### Diagramme de Dépendances

```
                    ┌─────────────┐
                    │    core/    │
                    │ (fondations)│
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ models/  │     │ analyse/ │     │ automl/  │
   │(ML eval) │     │ (stats)  │     │(training)│
   └────┬─────┘     └──────────┘     └──────────┘
        │
        ▼
┌───────────────────┐
│feature_engineering│
│   (LLMFE, DFS)    │
└───────────────────┘
        │
        └──────────────┐
                       ▼
               ┌──────────────┐
               │  pipeline/   │
               │(orchestrator)│
               └──────────────┘
                       │
                       ▼
               ┌──────────────┐
               │    front/    │
               │ (Streamlit)  │
               └──────────────┘
```

---

## Points d'Entrée du Code

### Pour comprendre le projet, commencer par :

| Si tu veux... | Commence par lire... |
|---------------|----------------------|
| Comprendre le flux complet | `src/pipeline/pipeline_all.py` → classe `FullPipeline` |
| Voir comment l'analyse fonctionne | `src/analyse/analyse.py` → fonction `analyse()` |
| Comprendre LLMFE | `src/feature_engineering/llmfe/llmfe_runner.py` |
| Voir comment les modèles sont évalués | `src/models/evaluation/cross_validator.py` |
| Ajouter un framework AutoML | `src/automl/supervised/` → voir les wrappers existants |

### Points d'entrée utilisateur :

| Interface | Fichier | Usage |
|-----------|---------|-------|
| **CLI** | `tests/integration/test_pipeline_all.py` | `python test_pipeline_all.py --dataset titanic --target Survived` |
| **Streamlit** | `src/front/pipeline_streamlit.py` | `streamlit run pipeline_streamlit.py` |
| **Python API** | `src/pipeline/pipeline_all.py` | `FullPipeline(project_name=...).run(...)` |

---

## Technologies Clés

### Machine Learning
| Composant | Technologie |
|-----------|-------------|
| Frameworks AutoML | FLAML, AutoGluon, TPOT, H2O |
| Modèles d'évaluation | XGBoost, LightGBM, CatBoost, RandomForest, sklearn |
| Feature Engineering | FeatureTools (DFS), Feature-Engine |

### LLM
| Composant | Technologie |
|-----------|-------------|
| Provider Cloud | OpenAI (gpt-4o, gpt-4o-mini, gpt-3.5-turbo) |
| Provider Local | Ollama (mistral, llama3, deepseek) |
| Client unifié | `src/core/llm_client.py` → `OllamaClient` |

### Infrastructure
| Composant | Technologie |
|-----------|-------------|
| Data | pandas, numpy, pyarrow (parquet) |
| Frontend | Streamlit |
| Configuration | python-dotenv, dataclasses |
| Tests | pytest |

---

## Structure des Outputs

Chaque exécution crée une structure cohérente :

```
outputs/{project_name}/
│
├── analyse/                          # Résultats de l'analyse
│   ├── stats/report_stats.json       # Stats pures
│   ├── full/report_full.json         # Enrichi LLM (si --with-llm)
│   └── agent_llm/conversation.json   # Historique conversation LLM
│
├── feature_engineering/              # Features générées
│   ├── features/
│   │   ├── train_fe.parquet
│   │   └── test_fe.parquet
│   └── llmfe/results/                # Évolution LLMFE
│
└── automl/                           # Modèles entraînés
    ├── flaml/
    ├── autogluon/
    ├── tpot/
    ├── h2o/
    └── results/comparison.json       # Comparaison des frameworks
```

---

## Conventions du Projet

### Nommage
- **snake_case** pour les fichiers, fonctions, variables
- **PascalCase** pour les classes
- **SCREAMING_SNAKE_CASE** pour les constantes

### Configuration
- Variables d'environnement dans `.env`
- Accès via `src/core/config.py` → singleton `settings`

### Gestion des chemins
- Chaque module a un `path_config.py` qui hérite de `BasePathConfig`
- Tous les outputs passent par ces classes

### LLM
- Toujours utiliser `OllamaClient` de `src/core/llm_client.py`
- Supporte OpenAI et Ollama avec la même interface

---

## Documentation Complémentaire

| Document | Description |
|----------|-------------|
| [MODULE_DEPENDENCIES.md](./MODULE_DEPENDENCIES.md) | Comment les modules s'utilisent entre eux |
| [docs/modules/core.md](../modules/core.md) | Documentation détaillée du module core |
| [docs/modules/analyse.md](../modules/analyse.md) | Documentation détaillée du module analyse |
| [README.md](../../README.md) | Guide utilisateur (CLI, installation) |
