# TODO - Implémentation Pipeline AutoML avec Méta-Learning

## Vue d'ensemble

```
Phase 0: Module Modèles Partagé      [████████████████████] 100% ✅ TERMINÉ
Phase 1: LLMFE Multi-Modèle          [████████████████████] 100% ✅ TERMINÉ
Phase 2: Meta-Learning Base          [░░░░░░░░░░░░░░░░░░░░] 0%   ← À FAIRE
Phase 3: Similarité & Recommandations[░░░░░░░░░░░░░░░░░░░░] 0%
Phase 4: Intégration Pipeline        [░░░░░░░░░░░░░░░░░░░░] 0%
Phase 5: Fonctionnalités Avancées    [░░░░░░░░░░░░░░░░░░░░] 0%
```

---

## Phase 0 : Module Modèles Partagé ✅ TERMINÉ

### Structure implémentée
```
src/models/
├── __init__.py           ✅
├── base.py               ✅ Interface BaseModel
├── registry.py           ✅ Registre centralisé
├── config.py             ✅ Configuration + métriques
├── wrappers/
│   ├── __init__.py       ✅
│   ├── xgboost_wrapper.py    ✅
│   ├── lightgbm_wrapper.py   ✅
│   ├── catboost_wrapper.py   ✅
│   └── sklearn_wrapper.py    ✅
└── evaluation/
    ├── __init__.py       ✅
    ├── cross_validator.py    ✅ Multi-modèle + pondération
    └── metrics.py            ✅ Métriques classification/régression
```

### Tests
- [x] `tests/unit/test_models_module.py` - Tests du module

---

## Phase 1 : LLMFE Multi-Modèle ✅ TERMINÉ

### Intégration réalisée
- [x] `src/feature_engineering/llmfe/model_evaluator.py` utilise `src.models`
- [x] Support multi-modèle (xgboost, lightgbm, randomforest)
- [x] Évaluation pondérée multi-métrique
- [x] Configuration via `EvaluationConfig`

### Tests
- [x] `tests/unit/test_eval_metric_config.py` - Config évaluation

---

## Phase 2 : Meta-Learning Base 🔴 À FAIRE

**Objectif** : Créer l'infrastructure pour stocker et récupérer les méta-features des datasets

### 2.1 Structure du module
- [ ] Créer la structure de dossiers
  ```
  src/meta_learning/
  ├── __init__.py
  ├── config.py
  ├── path_config.py
  ├── meta_features/
  │   ├── __init__.py
  │   ├── extractor.py        # Extraction depuis report_stats.json
  │   └── standardizer.py     # Normalisation des vecteurs
  └── registry/
      ├── __init__.py
      ├── database.py         # Stockage JSON
      └── models.py           # Dataclasses
  ```

### 2.2 Modèles de données
- [ ] `@dataclass MetaFeatureVector` - vecteur de méta-features
  - Dimensionnalité : n_rows, n_cols, n_numeric, n_categorical
  - Target : n_classes, class_imbalance_ratio
  - Qualité : missing_ratio
  - Distribution : mean_skewness, mean_kurtosis
- [ ] `@dataclass DatasetMetadata` - métadonnées complètes
- [ ] `@dataclass PipelineResult` - résultat d'exécution

### 2.3 Extracteur de méta-features
- [ ] `MetaFeatureExtractor.from_analysis_report(report_path)`
  - Parser le JSON de `report_stats.json` existant
- [ ] `MetaFeatureExtractor.to_vector()` - Conversion en numpy

### 2.4 Base de données JSON
- [ ] `MetaDatabase.register_dataset(name, meta_features)`
- [ ] `MetaDatabase.log_pipeline_result(dataset, framework, score)`
- [ ] `MetaDatabase.get_similar_datasets(mf_vector, top_k)`

### 2.5 Tests
- [ ] `tests/unit/test_meta_features.py`
- [ ] `tests/unit/test_meta_database.py`

---

## Phase 3 : Similarité & Recommandations 🔴 À FAIRE

**Objectif** : Trouver des datasets similaires et recommander des pipelines

### 3.1 Calcul de similarité
- [ ] `SimilarityCalculator.cosine(v1, v2)`
- [ ] `DatasetMatcher.find_similar(mf_vector, top_k=5)`

### 3.2 Recommandations
- [ ] `PipelineRecommender.recommend(similar_datasets)`
  - Retourne le framework dominant des datasets similaires
- [ ] `HPRecommender.get_warm_start(dataset, framework)`
  - Retourne les HP du meilleur run historique

### 3.3 Tests
- [ ] `tests/unit/test_similarity.py`
- [ ] `tests/unit/test_recommendations.py`

---

## Phase 4 : Intégration Pipeline 🔴 À FAIRE

**Objectif** : Connecter le méta-learning au pipeline existant

### 4.1 Hooks d'intégration
- [ ] `on_analysis_complete(project_name, report_path)`
  - Extraire méta-features
  - Chercher datasets similaires
  - Retourner recommandations
- [ ] `on_automl_complete(project_name, framework, score)`
  - Logger résultat dans MetaDatabase

### 4.2 Modification du pipeline
- [ ] Ajouter option `--use-meta-learning` au CLI
- [ ] Modifier `src/pipeline/pipeline_all.py`
  - Appeler hooks après analyse et AutoML
  - Afficher recommandations

### 4.3 Tests
- [ ] `tests/integration/test_meta_learning_pipeline.py`

---

## Phase 5 : Fonctionnalités Avancées 🔵 OPTIONNEL

### 5.1 Landmarking (méta-features rapides)
- [ ] Quick DecisionTree score
- [ ] Quick LogisticRegression score

### 5.2 Surrogate Model
- [ ] Prédire performance sans exécuter le pipeline

---

## Prochaines étapes recommandées

1. **Phase 2.1-2.2** : Créer la structure et les dataclasses
2. **Phase 2.3** : Extracteur depuis `report_stats.json`
3. **Phase 2.4** : Base de données JSON simple
4. **Phase 3** : Similarité cosine + recommandations
5. **Phase 4** : Intégration hooks dans le pipeline

---

## Notes

### Points d'attention
- Utiliser le `report_stats.json` existant comme source de méta-features
- Stockage JSON simple (pas de SQLite pour commencer)
- Option `--use-meta-learning` désactivée par défaut

### Dépendances existantes
Le projet utilise déjà :
- `numpy`, `pandas`, `scikit-learn`
- `xgboost`, `lightgbm`, `catboost`
- Pas besoin de nouvelles dépendances pour Phase 2-4
