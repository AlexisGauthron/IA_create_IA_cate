# Plan d'Architecture : Pipeline AutoML Ultime avec Vision Métier LLM + Méta-Learning

## Vision

Créer un **pipeline AutoML end-to-end** qui combine :
- L'automatisation complète (comme Auto-sklearn, H2O, TPOT)
- L'intelligence métier via LLM (unique différenciateur)
- L'approche multi-agents inspirée de [AutoML-Agent](https://arxiv.org/abs/2410.02958)
- **Le méta-learning de SML-AutoML** (recommandation de pipelines basée sur similarité de datasets)

---

## Architecture SML-AutoML Intégrée

### Concept Clé : Deux Phases (Offline/Online)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE OFFLINE (Base de Connaissance)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Pour chaque dataset historique D_i :                                       │
│                                                                             │
│  1. Extraire méta-features → D_meta = {mf₁, mf₂, ..., mfₙ}                  │
│     • n_rows, n_cols, n_classes, missing_ratio, class_imbalance            │
│     • feature_entropy, correlation_avg, skewness_mean, etc.                │
│                                                                             │
│  2. Tester K pipelines → E_matrix[i][k] = score du pipeline k sur D_i      │
│     • Pipeline = preprocessing + feature_engineering + model + HPO         │
│                                                                             │
│  3. Stocker dans la base de connaissance :                                  │
│     • D_meta: Caractéristiques des datasets                                │
│     • E_matrix: Matrice de performance (datasets × pipelines)              │
│     • P_library: Bibliothèque de pipelines pré-définis                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE ONLINE (Nouveau Dataset)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Nouveau dataset D_new                                               │
│                                                                             │
│  1. Extraire méta-features → mf_new                                         │
│                                                                             │
│  2. Calculer similarité avec D_meta existants                               │
│     similarity = cosine(mf_new, D_meta[i]) pour chaque dataset historique  │
│                                                                             │
│  3. SI max(similarity) ≥ seuil (ex: 0.85) :                                 │
│     → Utiliser le pipeline dominant du dataset similaire (warm-start)       │
│     → Réduire l'espace de recherche HPO                                     │
│                                                                             │
│  4. SINON :                                                                 │
│     → Lancer recherche complète CASH                                        │
│     → Ajouter résultats à la base de connaissance                          │
│                                                                             │
│  5. Stocker résultats pour enrichir la base (apprentissage continu)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Méta-Features Extraits (Standard)

| Catégorie | Méta-Features |
|-----------|---------------|
| **Dimensionnalité** | n_rows, n_cols, n_features, n_numeric, n_categorical |
| **Target** | n_classes, class_imbalance_ratio, minority_class_pct |
| **Qualité** | missing_ratio, duplicate_ratio, outlier_ratio |
| **Distribution** | mean_skewness, mean_kurtosis, feature_entropy_avg |
| **Corrélations** | avg_feature_correlation, max_target_correlation |
| **Landmarking** | quick_tree_score, quick_logreg_score (optionnel) |

---

## Architecture Cible

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATEUR PRINCIPAL                            │
│                         (Multi-Agent LLM Controller)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   AGENT 1     │           │   AGENT 2     │           │   AGENT 3     │
│   ANALYSE     │           │   FEATURE     │           │   MODÈLES     │
│   MÉTIER      │           │   ENGINEER    │           │   & HPO       │
└───────────────┘           └───────────────┘           └───────────────┘
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ • Comprendre  │           │ • Générer     │           │ • Sélection   │
│   le problème │           │   features    │           │   algorithme  │
│ • Contexte    │           │ • Transformer │           │ • HPO         │
│   métier      │           │ • Sélection   │           │ • Ensembles   │
│ • Recommander │           │ • Validation  │           │ • Stacking    │
│   métriques   │           │   multi-model │           │               │
└───────────────┘           └───────────────┘           └───────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────┐
                        │      AGENT 4          │
                        │   VÉRIFICATION &      │
                        │   EXPLICABILITÉ       │
                        └───────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────┐
                        │   RÉSULTAT FINAL      │
                        │   + Rapport Métier    │
                        └───────────────────────┘
```

---

## Pipeline Détaillé (7 Étapes)

### ÉTAPE 1 : Data Ingestion & Profiling
**Module** : `src/data_profiling/`

```
Input: CSV/DataFrame
    │
    ├─→ Détection automatique des types
    ├─→ Statistiques descriptives
    ├─→ Détection valeurs manquantes
    ├─→ Détection outliers
    ├─→ Analyse distributions
    └─→ Corrélations (Pearson, Spearman, MI, PhiK)

Output: DataProfile JSON
```

**Existant** : `src/analyse/statistiques/` ✅
**À améliorer** : Ajouter détection automatique du type de problème plus robuste

---

### ÉTAPE 2 : Analyse Métier LLM (UNIQUE)
**Module** : `src/business_agent/`

```
Input: DataProfile + Données échantillon
    │
    ├─→ Agent LLM analyse le contexte métier
    ├─→ Comprend la signification des features
    ├─→ Identifie les patterns métier importants
    ├─→ Recommande la métrique optimale
    ├─→ Suggère des features métier à créer
    └─→ Détecte les pièges potentiels (leakage, biais)

Output: BusinessInsights JSON
    {
        "problem_understanding": "...",
        "recommended_metric": "f1_weighted",
        "metric_justification": "...",
        "feature_suggestions": [...],
        "warnings": [...],
        "domain_knowledge": {...}
    }
```

**Existant** : `src/analyse/metier/` ✅
**À améliorer** :
- Structurer les outputs pour les passer aux autres agents
- Ajouter des prompts spécialisés par domaine (finance, santé, retail, etc.)

---

### ÉTAPE 3 : Data Preprocessing
**Module** : `src/preprocessing/`

```
Input: DataFrame + DataProfile + BusinessInsights
    │
    ├─→ Imputation valeurs manquantes
    │   └─→ Stratégie selon type (mean, median, mode, KNN, MICE)
    ├─→ Encodage catégoriel
    │   └─→ Label, OneHot, Target, Frequency selon cardinalité
    ├─→ Scaling numérique
    │   └─→ Standard, MinMax, Robust selon distribution
    ├─→ Gestion outliers
    │   └─→ Clip, Remove, Transform selon métier
    └─→ Création features temporelles (si dates)

Output: DataFrame prétraité + PreprocessingPipeline (sérialisable)
```

**Existant** : `src/core/preprocessing.py` (partiel)
**À créer** : Module dédié avec pipeline sklearn sérialisable

---

### ÉTAPE 4 : Feature Engineering Multi-Stratégie
**Module** : `src/feature_engineering/`

```
Input: DataFrame prétraité + BusinessInsights
    │
    ├─→ STRATÉGIE 1: Feature Engineering Classique
    │   └─→ Polynomiales, Interactions, Aggregations
    │
    ├─→ STRATÉGIE 2: LLMFE (existant)
    │   └─→ Génération guidée par LLM + évaluation multi-modèle
    │
    ├─→ STRATÉGIE 3: Deep Feature Synthesis (FeatureTools)
    │   └─→ Génération automatique si données relationnelles
    │
    └─→ STRATÉGIE 4: Features Métier (depuis BusinessInsights)
        └─→ Implémentation des suggestions du LLM métier

    │
    ▼
Feature Selection
    ├─→ Importance-based (RF, XGBoost)
    ├─→ Correlation filtering
    ├─→ Recursive Feature Elimination
    └─→ SHAP-based selection

Output: DataFrame enrichi + FeaturePipeline
```

**Existant** : `src/feature_engineering/llmfe/` ✅, `src/feature_engineering/libs/` ✅
**À améliorer** :
- Unifier les stratégies dans un orchestrateur
- Ajouter évaluation multi-modèle (pas seulement XGBoost)
- Implémenter feature selection automatique

---

### ÉTAPE 5 : Model Selection & Training (CASH)
**Module** : `src/model_selection/`

```
Input: DataFrame enrichi + BusinessInsights (métrique)
    │
    ├─→ PHASE 1: Quick Screening (5-10 modèles rapides)
    │   └─→ LogisticRegression, DecisionTree, RandomForest, XGBoost, LightGBM
    │   └─→ Évaluation rapide (1 fold ou petit time budget)
    │   └─→ Filtrage des top-K modèles
    │
    ├─→ PHASE 2: HPO sur Top-K modèles
    │   └─→ Bayesian Optimization (Optuna/SMAC)
    │   └─→ Ou Evolutionary (TPOT)
    │   └─→ Budget temps configurable
    │
    └─→ PHASE 3: AutoML Frameworks (optionnel)
        └─→ H2O AutoML
        └─→ AutoGluon
        └─→ FLAML

Output: Liste de modèles entraînés avec scores
```

**Existant** : `src/automl/` ✅
**À améliorer** :
- Ajouter phase de screening rapide
- Intégrer Optuna pour HPO custom
- Permettre de choisir entre HPO custom ou frameworks AutoML

---

### ÉTAPE 6 : Ensemble & Stacking
**Module** : `src/ensemble/`

```
Input: Top-N modèles entraînés
    │
    ├─→ STRATÉGIE 1: Voting (simple)
    │   └─→ Majority vote ou moyenne des probas
    │
    ├─→ STRATÉGIE 2: Weighted Average
    │   └─→ Poids optimisés sur validation
    │
    ├─→ STRATÉGIE 3: Stacking
    │   └─→ Meta-learner (LogReg, XGBoost) sur predictions OOF
    │
    └─→ STRATÉGIE 4: Blending
        └─→ Holdout set pour entraîner meta-learner

Output: Modèle ensemble final
```

**Existant** : H2O fait du stacking automatique
**À créer** : Module dédié pour ensembles custom

---

### ÉTAPE 7 : Validation & Explicabilité
**Module** : `src/validation/`

```
Input: Modèle final + Données test
    │
    ├─→ Métriques de performance
    │   └─→ Accuracy, F1, AUC, LogLoss, etc.
    │
    ├─→ Analyse des erreurs
    │   └─→ Confusion matrix, Classification report
    │   └─→ Cas mal classés (inspection)
    │
    ├─→ Explicabilité
    │   └─→ SHAP values (global + local)
    │   └─→ Feature importance
    │   └─→ Partial Dependence Plots
    │
    └─→ Rapport Métier LLM
        └─→ Interprétation des résultats en langage naturel
        └─→ Recommandations d'amélioration
        └─→ Alertes (overfitting, biais, etc.)

Output: Rapport complet (JSON + HTML + PDF)
```

**Existant** : Partiel dans les wrappers AutoML
**À créer** : Module unifié avec rapport LLM

---

## Structure de Fichiers Cible

```
src/
├── models/                          # NOUVEAU - Module partagé pour tous les modèles
│   ├── __init__.py
│   ├── base.py                      # BaseModel abstrait (interface commune)
│   ├── registry.py                  # Registre centralisé des modèles
│   ├── config.py                    # Configuration des modèles
│   │
│   ├── wrappers/                    # Wrappers uniformes pour chaque modèle
│   │   ├── __init__.py
│   │   ├── xgboost_wrapper.py       # XGBoostModel
│   │   ├── lightgbm_wrapper.py      # LightGBMModel
│   │   ├── catboost_wrapper.py      # CatBoostModel
│   │   ├── sklearn_wrapper.py       # RandomForest, DecisionTree, LogReg, etc.
│   │   └── h2o_wrapper.py           # H2OModel (optionnel, plus lent)
│   │
│   └── evaluation/                  # Évaluation unifiée
│       ├── __init__.py
│       ├── cross_validator.py       # CrossValidator (K-Fold unifié)
│       ├── metrics.py               # Métriques (accuracy, f1, auc, rmse, etc.)
│       └── scorer.py                # Scorer combiné (multi-métrique)
│
├── orchestrator/                    # NOUVEAU - Orchestrateur principal
│   ├── pipeline.py                  # Pipeline unifié
│   ├── config.py                    # Configuration globale
│   └── multi_agent.py               # Coordination agents LLM
│
├── meta_learning/                   # NOUVEAU - Module SML-AutoML
│   ├── __init__.py
│   ├── config.py                    # MetaLearningConfig
│   ├── path_config.py               # Extends BasePathConfig
│   │
│   ├── meta_features/               # Extraction méta-features
│   │   ├── __init__.py
│   │   ├── extractor.py             # MetaFeatureExtractor (depuis analyse JSON)
│   │   ├── statistics.py            # Calculs statistiques avancés
│   │   ├── landmarker.py            # Quick model scores (landmarking)
│   │   └── standardizer.py          # Normalisation des méta-features
│   │
│   ├── registry/                    # Base de connaissance (D_meta, E_matrix)
│   │   ├── __init__.py
│   │   ├── database.py              # MetaDatabase (JSON/SQLite)
│   │   ├── models.py                # DatasetMetadata, PipelineResult
│   │   ├── queries.py               # Requêtes de similarité
│   │   └── schema.py                # Schéma de données
│   │
│   ├── similarity/                  # Calcul de similarité
│   │   ├── __init__.py
│   │   ├── calculator.py            # SimilarityCalculator (cosine, euclidean)
│   │   └── matcher.py               # DatasetMatcher (trouve datasets similaires)
│   │
│   ├── recommendations/             # Recommandations de pipelines
│   │   ├── __init__.py
│   │   ├── pipeline_recommender.py  # Recommande pipeline basé sur similarité
│   │   ├── automl_recommender.py    # Recommande framework AutoML
│   │   ├── fe_recommender.py        # Recommande stratégie FE
│   │   └── hp_recommender.py        # Recommande hyperparamètres (warm-start)
│   │
│   ├── surrogate/                   # Modèle surrogate (optionnel)
│   │   ├── __init__.py
│   │   ├── trainer.py               # Entraîne surrogate sur E_matrix
│   │   └── predictor.py             # Prédit performance d'un pipeline
│   │
│   └── integration/                 # Hooks dans le pipeline existant
│       ├── __init__.py
│       ├── hooks.py                 # Points d'intégration
│       ├── enhance_runner.py        # AutoMLRunner avec warm-start
│       └── enhance_llmfe.py         # LLMFE avec recommandations
│
├── data_profiling/                  # RENOMMER depuis analyse/
│   ├── profiler.py                  # Profiling automatique
│   ├── statistics.py                # Stats descriptives
│   ├── correlations.py              # Corrélations avancées
│   └── leakage_detector.py          # Détection fuites
│
├── business_agent/                  # NOUVEAU - Agent métier LLM
│   ├── agent.py                     # Agent principal
│   ├── prompts/                     # Prompts par domaine
│   │   ├── generic.py
│   │   ├── finance.py
│   │   ├── healthcare.py
│   │   └── retail.py
│   └── insights.py                  # Structure BusinessInsights
│
├── preprocessing/                   # AMÉLIORER
│   ├── pipeline.py                  # Pipeline sklearn
│   ├── imputation.py                # Stratégies imputation
│   ├── encoding.py                  # Encodage catégoriel
│   ├── scaling.py                   # Normalisation
│   └── outliers.py                  # Gestion outliers
│
├── feature_engineering/             # AMÉLIORER
│   ├── orchestrator.py              # NOUVEAU - Orchestre les stratégies
│   ├── llmfe/                       # Existant
│   │   ├── llmfe_runner.py          # MODIFIER - Utilise src/models/
│   │   ├── evaluator.py             # MODIFIER - Utilise src/models/evaluation/
│   │   └── ...
│   ├── classical/                   # NOUVEAU
│   │   ├── interactions.py
│   │   ├── polynomials.py
│   │   └── aggregations.py
│   ├── selection/                   # NOUVEAU
│   │   ├── importance.py
│   │   ├── correlation.py
│   │   └── rfe.py
│   └── libs/                        # Existant
│
├── model_selection/                 # NOUVEAU - Utilise src/models/
│   ├── screening.py                 # Quick screening avec src/models/
│   ├── hpo.py                       # HPO avec Optuna + src/models/
│   └── cash.py                      # CASH utilise src/models/registry
│
├── automl/                          # GARDER - Frameworks externes
│   ├── runner.py
│   └── supervised/
│
├── ensemble/                        # NOUVEAU
│   ├── voting.py
│   ├── stacking.py
│   └── blending.py
│
├── validation/                      # NOUVEAU
│   ├── metrics.py
│   ├── error_analysis.py
│   ├── explainability.py            # SHAP, PDP
│   └── report_generator.py          # Rapport LLM
│
├── core/                            # GARDER
│   ├── config.py
│   ├── llm_client.py
│   └── ...
│
└── front/                           # GARDER
    └── ...
```

### Structure du Stockage Méta-Learning

```
outputs/
├── meta_learning_db/                # Base de connaissance globale
│   ├── registry_index.json          # Index de tous les datasets
│   ├── datasets/                    # Méta-features par dataset
│   │   ├── titanic_metadata.json
│   │   ├── iris_metadata.json
│   │   └── ...
│   ├── pipelines/                   # Résultats par pipeline
│   │   ├── titanic_h2o_result.json
│   │   ├── titanic_flaml_result.json
│   │   └── ...
│   ├── e_matrix.json                # Matrice de performance complète
│   └── meta_db.sqlite               # (Optionnel) Base SQLite
│
└── {project_name}/                  # Projet courant
    ├── analyse/
    ├── feature_engineering/
    └── automl/
```

---

## Flux d'Exécution Unifié avec Méta-Learning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INPUT: Dataset + Target                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. ANALYSE (src/analyse/) → report_stats.json                               │
│    • Extraire stats, distributions, corrélations                            │
│    • Générer insights métier via LLM                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. META-FEATURES (src/meta_learning/meta_features/)                         │
│    • Convertir report_stats.json → vecteur de méta-features normalisé       │
│    • Optionnel: Landmarking (quick model scores)                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. SIMILARITÉ (src/meta_learning/similarity/)                               │
│    • Chercher datasets similaires dans MetaDatabase                         │
│    • Calculer cosine similarity avec tous les datasets historiques          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
          similarity ≥ 0.85                   similarity < 0.85
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│ 5a. WARM-START (recommandations)  │   │ 5b. RECHERCHE COMPLÈTE (CASH)     │
│   • Pipeline dominant du dataset  │   │   • Tester toutes les stratégies  │
│     similaire                     │   │   • HPO complet                   │
│   • Hyperparamètres initiaux      │   │   • Exploration exhaustive        │
│   • Espace de recherche réduit    │   │                                   │
└───────────────────────────────────┘   └───────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. FEATURE ENGINEERING (src/feature_engineering/)                           │
│    • Stratégie recommandée (LLMFE, classical, ou les deux)                  │
│    • Évaluation multi-modèle (pas seulement XGBoost)                        │
│    • Feature selection automatique                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. MODEL SELECTION & HPO (src/model_selection/ + src/automl/)               │
│    • Quick screening → Top-K modèles                                        │
│    • HPO (Optuna) ou AutoML frameworks (H2O, FLAML, AutoGluon)             │
│    • Warm-start si dataset similaire trouvé                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. ENSEMBLE (src/ensemble/)                                                  │
│    • Voting, Stacking, ou Blending selon recommandation                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 9. VALIDATION & RAPPORT (src/validation/)                                    │
│    • Métriques finales, SHAP, rapport LLM                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 10. MISE À JOUR BASE DE CONNAISSANCE (src/meta_learning/registry/)          │
│     • Stocker méta-features du nouveau dataset                              │
│     • Stocker résultats des pipelines testés                                │
│     • Mettre à jour E_matrix pour apprentissage continu                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### API Unifiée

```python
from src.orchestrator import AutoMLPipeline

# Initialisation avec méta-learning
pipeline = AutoMLPipeline(
    project_name="my_project",
    llm_provider="openai",
    llm_model="gpt-4o",
    use_meta_learning=True,          # Active le méta-learning
    similarity_threshold=0.85         # Seuil pour warm-start
)

# Configuration (optionnel - sinon tout auto)
pipeline.configure(
    metric="auto",                    # Recommandé par LLM métier
    time_budget=3600,
    feature_strategies=["llmfe", "classical", "business"],
    evaluators=["xgboost", "lightgbm", "randomforest", "h2o"],  # Multi-modèle
    model_selection="auto",           # "screening+hpo" ou "automl_frameworks"
    ensemble_strategy="stacking",
    explain=True
)

# Exécution
result = pipeline.run(
    df_train=train_df,
    df_test=test_df,
    target_col="target"
)

# Résultats enrichis
print(result.best_score)
print(result.best_model)
print(result.business_report)
print(result.similar_datasets)        # Datasets similaires trouvés
print(result.recommended_pipeline)    # Pipeline recommandé
result.save("outputs/my_project/")
```

---

## Priorisation d'Implémentation

### Phase 1 : Foundation - LLMFE Multi-Modèle
**Objectif**: Permettre l'évaluation de features avec plusieurs modèles

| # | Tâche | Fichier | Priorité |
|---|-------|---------|----------|
| 1 | Créer évaluateurs modulaires | `src/feature_engineering/llmfe/evaluators.py` | 🔴 Critique |
| 2 | Modifier `_generate_spec()` pour séparer transform/evaluate | `src/feature_engineering/llmfe/llmfe_runner.py` | 🔴 Critique |
| 3 | Ajouter config pour sélectionner évaluateurs | `src/feature_engineering/llmfe/config.py` | 🟡 Important |

**Détail `evaluators.py`**:
```python
class BaseEvaluator(ABC):
    def evaluate(self, X, y, is_regression) -> float

class XGBoostEvaluator(BaseEvaluator): ...
class LightGBMEvaluator(BaseEvaluator): ...
class RandomForestEvaluator(BaseEvaluator): ...
class CatBoostEvaluator(BaseEvaluator): ...
class DecisionTreeEvaluator(BaseEvaluator): ...  # Simple, évite overfitting
class H2OAutoMLEvaluator(BaseEvaluator): ...     # Optionnel, plus lent

EVALUATORS = {"xgboost": XGBoostEvaluator, ...}

def get_evaluator(name: str) -> BaseEvaluator
def get_multi_evaluator(names: list[str]) -> MultiModelEvaluator  # Moyenne/vote
```

### Phase 2 : Meta-Learning - Base de Connaissance
**Objectif**: Stocker et récupérer méta-features des datasets

| # | Tâche | Fichier | Priorité |
|---|-------|---------|----------|
| 4 | Créer extracteur de méta-features | `src/meta_learning/meta_features/extractor.py` | 🔴 Critique |
| 5 | Créer modèles de données | `src/meta_learning/registry/models.py` | 🔴 Critique |
| 6 | Créer base de données JSON | `src/meta_learning/registry/database.py` | 🔴 Critique |
| 7 | Créer normaliseur | `src/meta_learning/meta_features/standardizer.py` | 🟡 Important |

**Détail `extractor.py`**:
```python
@dataclass
class MetaFeatureVector:
    """Vecteur de 30+ méta-features normalisés"""
    # Dimensionnalité
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    feature_ratio: float  # n_features / n_rows

    # Target
    n_classes: int  # 0 si régression
    class_imbalance_ratio: float
    minority_class_pct: float

    # Qualité
    missing_ratio: float
    duplicate_ratio: float

    # Distribution
    mean_skewness: float
    mean_kurtosis: float
    mean_entropy: float

    # Corrélations
    avg_feature_correlation: float
    max_target_correlation: float

class MetaFeatureExtractor:
    def from_analysis_report(self, report_path: str) -> MetaFeatureVector
    def from_dataframe(self, df: pd.DataFrame, target: str) -> MetaFeatureVector
    def to_vector(self, mf: MetaFeatureVector) -> np.ndarray  # Pour similarité
```

### Phase 3 : Meta-Learning - Similarité & Recommandations
**Objectif**: Trouver datasets similaires et recommander pipelines

| # | Tâche | Fichier | Priorité |
|---|-------|---------|----------|
| 8 | Créer calculateur de similarité | `src/meta_learning/similarity/calculator.py` | 🔴 Critique |
| 9 | Créer matcher de datasets | `src/meta_learning/similarity/matcher.py` | 🔴 Critique |
| 10 | Créer recommandeur de pipeline | `src/meta_learning/recommendations/pipeline_recommender.py` | 🟡 Important |
| 11 | Créer recommandeur FE | `src/meta_learning/recommendations/fe_recommender.py` | 🟡 Important |

**Détail `matcher.py`**:
```python
class DatasetMatcher:
    def __init__(self, db: MetaDatabase, similarity_threshold: float = 0.85):
        self.db = db
        self.threshold = similarity_threshold

    def find_similar(self, mf_vector: np.ndarray, top_k: int = 5) -> list[SimilarDataset]:
        """Retourne les k datasets les plus similaires avec score >= threshold"""

    def get_dominant_pipeline(self, dataset_name: str) -> PipelineConfig:
        """Retourne le meilleur pipeline pour un dataset donné"""
```

### Phase 4 : Integration - Hooks dans Pipeline Existant
**Objectif**: Connecter méta-learning au pipeline Analyse → FE → AutoML

| # | Tâche | Fichier | Priorité |
|---|-------|---------|----------|
| 12 | Créer hooks d'intégration | `src/meta_learning/integration/hooks.py` | 🟡 Important |
| 13 | Améliorer LLMFE avec warm-start | `src/meta_learning/integration/enhance_llmfe.py` | 🟢 Nice-to-have |
| 14 | Améliorer AutoMLRunner | `src/meta_learning/integration/enhance_runner.py` | 🟢 Nice-to-have |

**Points d'intégration**:
```python
# Hook 1: Après analyse (src/pipeline/pipeline_all.py)
meta_features = MetaFeatureExtractor().from_analysis_report(report_path)
MetaDatabase().register_dataset(project_name, meta_features)
similar = DatasetMatcher().find_similar(meta_features.to_vector())

# Hook 2: Avant LLMFE (si dataset similaire trouvé)
if similar and similar[0].score >= 0.85:
    recommended_fe = FERecommender().get_best_strategy(similar[0].name)
    # Utiliser stratégie recommandée ou warm-start

# Hook 3: Avant AutoML
if similar:
    best_hp = HPRecommender().get_warm_start(similar[0].name, framework="h2o")
    # Initialiser HPO avec ces hyperparamètres

# Hook 4: Après AutoML (toujours)
MetaDatabase().log_result(project_name, framework, score, hyperparameters)
```

### Phase 5 : Orchestrateur Unifié
**Objectif**: Pipeline complet avec API simple

| # | Tâche | Fichier | Priorité |
|---|-------|---------|----------|
| 15 | Créer orchestrateur principal | `src/orchestrator/pipeline.py` | 🟢 Nice-to-have |
| 16 | Créer config globale | `src/orchestrator/config.py` | 🟢 Nice-to-have |
| 17 | Créer module validation | `src/validation/` | 🟢 Nice-to-have |

### Phase 6 : Avancé (Optionnel)
**Objectif**: Fonctionnalités avancées SML-AutoML

| # | Tâche | Description | Priorité |
|---|-------|-------------|----------|
| 18 | Landmarking | Quick model scores pour méta-features | 🔵 Optionnel |
| 19 | Surrogate model | Prédire performance sans exécuter | 🔵 Optionnel |
| 20 | Transfer learning | Transférer features entre datasets | 🔵 Optionnel |

---

## Différenciateurs vs AutoML Classique

| Aspect | AutoML Classique | SML-AutoML | Notre Pipeline |
|--------|------------------|------------|----------------|
| **Compréhension métier** | Aucune | Aucune | Agent LLM dédié ✨ |
| **Méta-learning** | Non | Similarité-based | Similarité + LLM ✨ |
| **Cold-start** | Recherche exhaustive | Warm-start si similaire | Warm-start + recommandations LLM |
| **Feature Engineering** | Auto ou manuel | Pipeline library | Multi-stratégie + LLM + LLMFE ✨ |
| **Choix métrique** | Manuel | Basé sur méta-features | Recommandé par LLM métier ✨ |
| **Explicabilité** | SHAP basique | Non | Rapport en langage naturel ✨ |
| **Warnings** | Techniques | Techniques | Métier (biais, leakage, sens business) ✨ |
| **Évaluation FE** | 1 modèle | 1 modèle | Multi-modèle configurable ✨ |
| **Apprentissage continu** | Non | Oui (E_matrix) | Oui (E_matrix + feedback LLM) ✨ |

### Notre Valeur Ajoutée Unique

1. **LLM Business Agent** : Comprend le contexte métier, recommande la métrique, détecte les pièges
2. **LLMFE Multi-Modèle** : Feature engineering guidé par LLM, évalué sur plusieurs modèles
3. **Méta-Learning Hybride** : Combine similarité de datasets + intelligence LLM pour recommandations
4. **Rapport Métier** : Explications en langage naturel, pas seulement SHAP values

---

## Sources

- [SML-AutoML: Similarity-based Meta-Learning](https://www.oajaiml.com/uploads/archivepdf/134544176.pdf) - Framework méta-learning avec phases offline/online
- [AutoML-Agent: A Multi-Agent LLM Framework](https://arxiv.org/abs/2410.02958) - Architecture multi-agents avec LLM
- [H2O AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [Automated Feature Engineering Survey](https://arxiv.org/abs/2403.11395)
- [Google Vertex AI AutoML](https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/e2e-automl)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/) - Meta-learning avec warm-start

---

## Résumé Exécutif

Ce pipeline combine **trois innovations majeures** :

1. **SML-AutoML** (Méta-learning) :
   - Phase offline : Construire base de connaissance (D_meta, E_matrix)
   - Phase online : Trouver datasets similaires → recommander pipeline
   - Warm-start : Réduire temps de recherche si dataset similaire trouvé

2. **Vision Métier LLM** (Unique) :
   - Agent LLM analyse le contexte business
   - Recommande métrique optimale avec justification
   - Génère rapport explicatif en langage naturel

3. **LLMFE Multi-Modèle** :
   - Feature engineering guidé par LLM
   - Évaluation sur XGBoost, LightGBM, RandomForest, etc.
   - Évite l'overfitting à un seul modèle

**Résultat** : Un AutoML qui apprend de ses expériences passées ET comprend le contexte métier.
