# TODO - Implémentation Pipeline AutoML avec Méta-Learning

## Vue d'ensemble

```
Phase 0: Module Modèles Partagé      [░░░░░░░░░░░░░░░░░░░░] 0%  ← FONDATION
Phase 1: LLMFE Multi-Modèle          [░░░░░░░░░░░░░░░░░░░░] 0%
Phase 2: Meta-Learning Base          [░░░░░░░░░░░░░░░░░░░░] 0%
Phase 3: Similarité & Recommandations[░░░░░░░░░░░░░░░░░░░░] 0%
Phase 4: Intégration Pipeline        [░░░░░░░░░░░░░░░░░░░░] 0%
Phase 5: Orchestrateur Unifié        [░░░░░░░░░░░░░░░░░░░░] 0%
Phase 6: Fonctionnalités Avancées    [░░░░░░░░░░░░░░░░░░░░] 0%
```

---

## Phase 0 : Module Modèles Partagé 🔴 FONDATION

**Objectif** : Créer un module centralisé pour tous les modèles ML, réutilisable par LLMFE, CASH, screening, HPO

### 0.1 Structure du module
- [ ] **0.1.1** Créer la structure de dossiers
  ```
  src/models/
  ├── __init__.py
  ├── base.py                      # Interface BaseModel
  ├── registry.py                  # Registre centralisé
  ├── config.py                    # Configuration
  ├── wrappers/
  │   ├── __init__.py
  │   ├── xgboost_wrapper.py
  │   ├── lightgbm_wrapper.py
  │   ├── catboost_wrapper.py
  │   └── sklearn_wrapper.py
  └── evaluation/
      ├── __init__.py
      ├── cross_validator.py
      ├── metrics.py
      └── scorer.py
  ```

### 0.2 Interface BaseModel
- [ ] **0.2.1** Créer `src/models/base.py`
  ```python
  from abc import ABC, abstractmethod
  from typing import Any
  import numpy as np
  import pandas as pd

  class BaseModel(ABC):
      """Interface commune pour tous les modèles ML"""

      def __init__(self, is_regression: bool = False, random_state: int = 42, **kwargs):
          self.is_regression = is_regression
          self.random_state = random_state
          self.params = kwargs
          self.model = None

      @abstractmethod
      def get_name(self) -> str:
          """Nom du modèle (ex: 'xgboost', 'lightgbm')"""
          pass

      @abstractmethod
      def create_model(self, **hp) -> Any:
          """Crée une instance du modèle avec hyperparamètres"""
          pass

      @abstractmethod
      def get_default_params(self) -> dict:
          """Hyperparamètres par défaut"""
          pass

      @abstractmethod
      def get_hp_space(self) -> dict:
          """Espace de recherche des hyperparamètres (pour HPO)"""
          pass

      def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":
          params = {**self.get_default_params(), **self.params}
          self.model = self.create_model(**params)
          self.model.fit(X, y)
          return self

      def predict(self, X: pd.DataFrame) -> np.ndarray:
          return self.model.predict(X)

      def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
          if hasattr(self.model, 'predict_proba'):
              return self.model.predict_proba(X)
          raise NotImplementedError(f"{self.get_name()} doesn't support predict_proba")

      def clone(self) -> "BaseModel":
          """Crée une copie du modèle (non entraîné)"""
          return self.__class__(
              is_regression=self.is_regression,
              random_state=self.random_state,
              **self.params
          )
  ```

### 0.3 Wrappers des modèles
- [ ] **0.3.1** Créer `src/models/wrappers/xgboost_wrapper.py`
  ```python
  import xgboost as xgb
  from src.models.base import BaseModel

  class XGBoostModel(BaseModel):
      def get_name(self) -> str:
          return "xgboost"

      def get_default_params(self) -> dict:
          return {
              "n_estimators": 100,
              "max_depth": 6,
              "learning_rate": 0.1,
              "random_state": self.random_state,
              "n_jobs": -1,
              "verbosity": 0,
          }

      def get_hp_space(self) -> dict:
          return {
              "n_estimators": (50, 500),
              "max_depth": (3, 10),
              "learning_rate": (0.01, 0.3),
              "subsample": (0.6, 1.0),
              "colsample_bytree": (0.6, 1.0),
          }

      def create_model(self, **hp):
          if self.is_regression:
              return xgb.XGBRegressor(**hp)
          return xgb.XGBClassifier(**hp)
  ```

- [ ] **0.3.2** Créer `src/models/wrappers/lightgbm_wrapper.py`
  ```python
  import lightgbm as lgb
  from src.models.base import BaseModel

  class LightGBMModel(BaseModel):
      def get_name(self) -> str:
          return "lightgbm"

      def get_default_params(self) -> dict:
          return {
              "n_estimators": 100,
              "max_depth": -1,
              "learning_rate": 0.1,
              "num_leaves": 31,
              "random_state": self.random_state,
              "n_jobs": -1,
              "verbose": -1,
          }

      def get_hp_space(self) -> dict:
          return {
              "n_estimators": (50, 500),
              "max_depth": (3, 15),
              "learning_rate": (0.01, 0.3),
              "num_leaves": (20, 100),
              "subsample": (0.6, 1.0),
          }

      def create_model(self, **hp):
          if self.is_regression:
              return lgb.LGBMRegressor(**hp)
          return lgb.LGBMClassifier(**hp)
  ```

- [ ] **0.3.3** Créer `src/models/wrappers/sklearn_wrapper.py`
  ```python
  from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
  from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
  from sklearn.linear_model import LogisticRegression, Ridge
  from src.models.base import BaseModel

  class RandomForestModel(BaseModel):
      def get_name(self) -> str:
          return "randomforest"

      def get_default_params(self) -> dict:
          return {
              "n_estimators": 100,
              "max_depth": None,
              "random_state": self.random_state,
              "n_jobs": -1,
          }

      def get_hp_space(self) -> dict:
          return {
              "n_estimators": (50, 300),
              "max_depth": (5, 20),
              "min_samples_split": (2, 10),
          }

      def create_model(self, **hp):
          if self.is_regression:
              return RandomForestRegressor(**hp)
          return RandomForestClassifier(**hp)


  class DecisionTreeModel(BaseModel):
      def get_name(self) -> str:
          return "decisiontree"

      def get_default_params(self) -> dict:
          return {
              "max_depth": 5,
              "random_state": self.random_state,
          }

      def get_hp_space(self) -> dict:
          return {
              "max_depth": (3, 15),
              "min_samples_split": (2, 20),
          }

      def create_model(self, **hp):
          if self.is_regression:
              return DecisionTreeRegressor(**hp)
          return DecisionTreeClassifier(**hp)


  class LogisticRegressionModel(BaseModel):
      def get_name(self) -> str:
          return "logistic"

      def get_default_params(self) -> dict:
          return {
              "max_iter": 1000,
              "random_state": self.random_state,
              "n_jobs": -1,
          }

      def get_hp_space(self) -> dict:
          return {
              "C": (0.01, 10.0),
              "penalty": ["l1", "l2"],
          }

      def create_model(self, **hp):
          if self.is_regression:
              return Ridge(random_state=self.random_state)
          return LogisticRegression(**hp)
  ```

- [ ] **0.3.4** Créer `src/models/wrappers/catboost_wrapper.py`
  ```python
  from catboost import CatBoostClassifier, CatBoostRegressor
  from src.models.base import BaseModel

  class CatBoostModel(BaseModel):
      def get_name(self) -> str:
          return "catboost"

      def get_default_params(self) -> dict:
          return {
              "iterations": 100,
              "depth": 6,
              "learning_rate": 0.1,
              "random_seed": self.random_state,
              "verbose": False,
          }

      def get_hp_space(self) -> dict:
          return {
              "iterations": (50, 500),
              "depth": (4, 10),
              "learning_rate": (0.01, 0.3),
          }

      def create_model(self, **hp):
          if self.is_regression:
              return CatBoostRegressor(**hp)
          return CatBoostClassifier(**hp)
  ```

### 0.4 Registre centralisé
- [ ] **0.4.1** Créer `src/models/registry.py`
  ```python
  from typing import Type
  from src.models.base import BaseModel
  from src.models.wrappers.xgboost_wrapper import XGBoostModel
  from src.models.wrappers.lightgbm_wrapper import LightGBMModel
  from src.models.wrappers.catboost_wrapper import CatBoostModel
  from src.models.wrappers.sklearn_wrapper import (
      RandomForestModel, DecisionTreeModel, LogisticRegressionModel
  )

  MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
      "xgboost": XGBoostModel,
      "lightgbm": LightGBMModel,
      "catboost": CatBoostModel,
      "randomforest": RandomForestModel,
      "decisiontree": DecisionTreeModel,
      "logistic": LogisticRegressionModel,
  }

  def get_model(name: str, is_regression: bool = False, **kwargs) -> BaseModel:
      """Récupère un modèle par son nom"""
      if name not in MODEL_REGISTRY:
          available = list(MODEL_REGISTRY.keys())
          raise ValueError(f"Unknown model: {name}. Available: {available}")
      return MODEL_REGISTRY[name](is_regression=is_regression, **kwargs)

  def get_models(names: list[str], is_regression: bool = False) -> list[BaseModel]:
      """Récupère plusieurs modèles"""
      return [get_model(name, is_regression) for name in names]

  def get_all_models(is_regression: bool = False) -> list[BaseModel]:
      """Récupère tous les modèles disponibles"""
      return [cls(is_regression=is_regression) for cls in MODEL_REGISTRY.values()]

  def list_models() -> list[str]:
      """Liste les noms de modèles disponibles"""
      return list(MODEL_REGISTRY.keys())

  def register_model(name: str, model_class: Type[BaseModel]) -> None:
      """Enregistre un nouveau modèle"""
      MODEL_REGISTRY[name] = model_class
  ```

### 0.5 Module d'évaluation
- [ ] **0.5.1** Créer `src/models/evaluation/metrics.py`
  ```python
  from sklearn.metrics import (
      accuracy_score, f1_score, roc_auc_score, log_loss,
      mean_squared_error, mean_absolute_error, r2_score
  )
  import numpy as np

  CLASSIFICATION_METRICS = {
      "accuracy": accuracy_score,
      "f1": lambda y, p: f1_score(y, p, average="weighted"),
      "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
      "auc": roc_auc_score,
      "logloss": log_loss,
  }

  REGRESSION_METRICS = {
      "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
      "mse": mean_squared_error,
      "mae": mean_absolute_error,
      "r2": r2_score,
  }

  def get_metric(name: str, is_regression: bool):
      metrics = REGRESSION_METRICS if is_regression else CLASSIFICATION_METRICS
      if name == "auto":
          return metrics["rmse"] if is_regression else metrics["f1"]
      return metrics[name]

  def get_default_metric(is_regression: bool) -> str:
      return "rmse" if is_regression else "f1"
  ```

- [ ] **0.5.2** Créer `src/models/evaluation/cross_validator.py`
  ```python
  from sklearn.model_selection import StratifiedKFold, KFold
  import numpy as np
  import pandas as pd
  from src.models.base import BaseModel
  from src.models.evaluation.metrics import get_metric, get_default_metric

  class CrossValidator:
      """Validation croisée unifiée pour tous les modèles"""

      def __init__(self, n_folds: int = 5, shuffle: bool = True, random_state: int = 42):
          self.n_folds = n_folds
          self.shuffle = shuffle
          self.random_state = random_state

      def evaluate(
          self,
          model: BaseModel,
          X: pd.DataFrame,
          y: np.ndarray,
          metric: str = "auto"
      ) -> dict:
          """Évalue un modèle en cross-validation"""
          is_regression = model.is_regression

          if is_regression:
              kfold = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
          else:
              kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

          metric_fn = get_metric(metric, is_regression)
          scores = []

          for train_idx, val_idx in kfold.split(X, y):
              X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
              y_train, y_val = y[train_idx], y[val_idx]

              model_clone = model.clone()
              model_clone.fit(X_train, y_train)
              y_pred = model_clone.predict(X_val)

              score = metric_fn(y_val, y_pred)
              scores.append(score)

          return {
              "mean": np.mean(scores),
              "std": np.std(scores),
              "scores": scores,
              "metric": metric if metric != "auto" else get_default_metric(is_regression)
          }

      def evaluate_multi_model(
          self,
          models: list[BaseModel],
          X: pd.DataFrame,
          y: np.ndarray,
          metric: str = "auto",
          aggregation: str = "mean"
      ) -> dict:
          """Évalue plusieurs modèles et agrège les scores"""
          results = {}
          all_means = []

          for model in models:
              result = self.evaluate(model, X, y, metric)
              results[model.get_name()] = result
              all_means.append(result["mean"])

          # Agrégation
          if aggregation == "mean":
              final_score = np.mean(all_means)
          elif aggregation == "min":
              final_score = np.min(all_means)
          elif aggregation == "max":
              final_score = np.max(all_means)
          else:
              final_score = np.mean(all_means)

          return {
              "aggregated_score": final_score,
              "aggregation": aggregation,
              "individual_results": results
          }
  ```

### 0.6 Configuration
- [ ] **0.6.1** Créer `src/models/config.py`
  ```python
  from dataclasses import dataclass, field

  @dataclass
  class ModelsConfig:
      """Configuration pour le module models"""
      default_models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "randomforest"])
      n_folds: int = 5
      random_state: int = 42
      metric: str = "auto"
      aggregation: str = "mean"  # mean, min, max
  ```

### 0.7 Tests
- [ ] **0.7.1** Créer `tests/unit/test_models/test_base.py`
  - [ ] Test interface BaseModel
  - [ ] Test clone()

- [ ] **0.7.2** Créer `tests/unit/test_models/test_wrappers.py`
  - [ ] Test chaque wrapper (XGBoost, LightGBM, CatBoost, sklearn)
  - [ ] Test classification et régression

- [ ] **0.7.3** Créer `tests/unit/test_models/test_registry.py`
  - [ ] Test get_model()
  - [ ] Test get_all_models()
  - [ ] Test register_model()

- [ ] **0.7.4** Créer `tests/unit/test_models/test_cross_validator.py`
  - [ ] Test evaluate() single model
  - [ ] Test evaluate_multi_model()
  - [ ] Test différentes métriques

---

## Phase 1 : LLMFE Multi-Modèle 🔴 CRITIQUE (utilise src/models/)

**Objectif** : Modifier LLMFE pour utiliser le module modèles partagé

### 1.1 Modifier l'évaluateur LLMFE
- [ ] **1.1.1** Modifier `src/feature_engineering/llmfe/evaluator.py`
  - [ ] Importer `from src.models.registry import get_model, get_models`
  - [ ] Importer `from src.models.evaluation import CrossValidator`
  - [ ] Remplacer le code XGBoost hardcodé par appel au registre
  - [ ] Ajouter support multi-modèle

  ```python
  from src.models.registry import get_models
  from src.models.evaluation import CrossValidator

  class LLMFEEvaluator:
      def __init__(
          self,
          model_names: list[str] = ["xgboost"],
          n_folds: int = 4,
          aggregation: str = "mean"
      ):
          self.model_names = model_names
          self.n_folds = n_folds
          self.aggregation = aggregation
          self.cv = CrossValidator(n_folds=n_folds)

      def evaluate(self, X: pd.DataFrame, y: np.ndarray, is_regression: bool) -> float:
          models = get_models(self.model_names, is_regression)
          result = self.cv.evaluate_multi_model(
              models, X, y, aggregation=self.aggregation
          )
          return result["aggregated_score"]
  ```

### 1.2 Modifier LLMFE Runner
- [ ] **1.2.1** Modifier `src/feature_engineering/llmfe/llmfe_runner.py`
  - [ ] Séparer génération de `modify_features()` de l'évaluation
  - [ ] Supprimer le code XGBoost du prompt envoyé au LLM
  - [ ] Ajouter paramètre `model_names: list[str]` dans la config
  - [ ] Modifier `_generate_spec()` pour ne générer que la transformation

### 1.3 Modifier la configuration LLMFE
- [ ] **1.3.1** Mettre à jour `src/feature_engineering/llmfe/config.py`
  - [ ] Ajouter `model_names: list[str] = ["xgboost"]`
  - [ ] Ajouter `aggregation: str = "mean"`
  - [ ] Garder rétrocompatibilité

### 1.4 Tests
- [ ] **1.4.1** Créer `tests/unit/test_llmfe_evaluator.py`
  - [ ] Test avec un seul modèle
  - [ ] Test avec plusieurs modèles
  - [ ] Test aggregation (mean, min, max)

- [ ] **1.4.2** Test d'intégration LLMFE
  - [ ] Vérifier que le score final est cohérent
  - [ ] Comparer avec l'ancien comportement

---

## Phase 2 : Meta-Learning - Base de Connaissance 🔴 CRITIQUE

**Objectif** : Créer l'infrastructure pour stocker et récupérer les méta-features des datasets

### 2.1 Structure du module
- [ ] **2.1.1** Créer la structure de dossiers
  ```
  src/meta_learning/
  ├── __init__.py
  ├── config.py
  ├── path_config.py
  ├── meta_features/
  │   ├── __init__.py
  │   ├── extractor.py
  │   ├── statistics.py
  │   └── standardizer.py
  └── registry/
      ├── __init__.py
      ├── database.py
      ├── models.py
      └── queries.py
  ```

### 2.2 Modèles de données
- [ ] **2.2.1** Créer `src/meta_learning/registry/models.py`
  - [ ] `@dataclass MetaFeatureVector` - vecteur de méta-features
    - Dimensionnalité : n_rows, n_cols, n_numeric, n_categorical, feature_ratio
    - Target : n_classes, class_imbalance_ratio, minority_class_pct
    - Qualité : missing_ratio, duplicate_ratio
    - Distribution : mean_skewness, mean_kurtosis, mean_entropy
    - Corrélations : avg_feature_correlation, max_target_correlation
  - [ ] `@dataclass DatasetMetadata` - métadonnées complètes d'un dataset
  - [ ] `@dataclass PipelineResult` - résultat d'exécution d'un pipeline
  - [ ] `@dataclass SimilarDataset` - dataset similaire avec score

### 2.3 Extracteur de méta-features
- [ ] **2.3.1** Créer `src/meta_learning/meta_features/extractor.py`
  - [ ] `MetaFeatureExtractor.from_analysis_report(report_path: str) -> MetaFeatureVector`
    - Parser le JSON de `report_stats.json` existant
    - Extraire tous les méta-features
  - [ ] `MetaFeatureExtractor.from_dataframe(df, target) -> MetaFeatureVector`
    - Calcul direct depuis DataFrame (fallback)
  - [ ] `MetaFeatureExtractor.to_vector(mf: MetaFeatureVector) -> np.ndarray`
    - Conversion en vecteur numpy pour calcul de similarité

- [ ] **2.3.2** Créer `src/meta_learning/meta_features/standardizer.py`
  - [ ] `MetaFeatureStandardizer.fit(vectors: list[np.ndarray])`
  - [ ] `MetaFeatureStandardizer.transform(vector: np.ndarray) -> np.ndarray`
  - [ ] Sauvegarder/charger le standardizer (pickle ou JSON)

### 2.4 Base de données
- [ ] **2.4.1** Créer `src/meta_learning/registry/database.py`
  - [ ] `MetaDatabase.__init__(db_path: str)` - initialisation
  - [ ] `MetaDatabase.register_dataset(name, meta_features, metadata)` - ajouter dataset
  - [ ] `MetaDatabase.get_dataset(name) -> DatasetMetadata` - récupérer dataset
  - [ ] `MetaDatabase.get_all_datasets() -> list[DatasetMetadata]` - tous les datasets
  - [ ] `MetaDatabase.log_pipeline_result(dataset, framework, score, hp)` - logger résultat
  - [ ] `MetaDatabase.get_pipeline_results(dataset) -> list[PipelineResult]` - résultats
  - [ ] `MetaDatabase.get_e_matrix() -> pd.DataFrame` - matrice de performance

- [ ] **2.4.2** Format de stockage JSON
  ```
  outputs/meta_learning_db/
  ├── registry_index.json      # Index des datasets
  ├── datasets/
  │   └── {name}_metadata.json # Méta-features par dataset
  ├── pipelines/
  │   └── {name}_{framework}_result.json
  └── e_matrix.json            # Matrice de performance
  ```

### 2.5 Configuration
- [ ] **2.5.1** Créer `src/meta_learning/config.py`
  - [ ] `MetaLearningConfig` avec paramètres par défaut
  - [ ] Path vers la base de données

- [ ] **2.5.2** Créer `src/meta_learning/path_config.py`
  - [ ] Étendre `BasePathConfig` existant
  - [ ] Définir chemins pour meta_learning_db

### 2.6 Tests
- [ ] **2.6.1** Créer `tests/unit/test_meta_features.py`
  - [ ] Test extraction depuis report_stats.json
  - [ ] Test extraction depuis DataFrame
  - [ ] Test standardization

- [ ] **2.6.2** Créer `tests/unit/test_meta_database.py`
  - [ ] Test CRUD datasets
  - [ ] Test logging résultats
  - [ ] Test E_matrix

---

## Phase 3 : Similarité & Recommandations 🔴 CRITIQUE

**Objectif** : Trouver des datasets similaires et recommander des pipelines

### 3.1 Structure du module
- [ ] **3.1.1** Créer la structure
  ```
  src/meta_learning/
  ├── similarity/
  │   ├── __init__.py
  │   ├── calculator.py
  │   └── matcher.py
  └── recommendations/
      ├── __init__.py
      ├── pipeline_recommender.py
      ├── fe_recommender.py
      └── hp_recommender.py
  ```

### 3.2 Calcul de similarité
- [ ] **3.2.1** Créer `src/meta_learning/similarity/calculator.py`
  - [ ] `SimilarityCalculator.cosine(v1, v2) -> float`
  - [ ] `SimilarityCalculator.euclidean(v1, v2) -> float`
  - [ ] `SimilarityCalculator.compute(v1, v2, method="cosine") -> float`

- [ ] **3.2.2** Créer `src/meta_learning/similarity/matcher.py`
  - [ ] `DatasetMatcher.__init__(db: MetaDatabase, threshold: float = 0.85)`
  - [ ] `DatasetMatcher.find_similar(mf_vector, top_k=5) -> list[SimilarDataset]`
  - [ ] `DatasetMatcher.get_dominant_pipeline(dataset_name) -> PipelineConfig`
  - [ ] `DatasetMatcher.is_similar_enough(score) -> bool`

### 3.3 Recommandations
- [ ] **3.3.1** Créer `src/meta_learning/recommendations/pipeline_recommender.py`
  - [ ] `PipelineRecommender.recommend(similar_datasets) -> PipelineConfig`
  - [ ] Logique : prendre le pipeline dominant du dataset le plus similaire

- [ ] **3.3.2** Créer `src/meta_learning/recommendations/fe_recommender.py`
  - [ ] `FERecommender.recommend(similar_datasets) -> FEStrategy`
  - [ ] Retourne : "llmfe", "classical", "both", "none"

- [ ] **3.3.3** Créer `src/meta_learning/recommendations/hp_recommender.py`
  - [ ] `HPRecommender.get_warm_start(dataset, framework) -> dict`
  - [ ] Retourne les hyperparamètres du meilleur run historique

### 3.4 Tests
- [ ] **3.4.1** Créer `tests/unit/test_similarity.py`
  - [ ] Test calcul cosine/euclidean
  - [ ] Test find_similar avec données mock

- [ ] **3.4.2** Créer `tests/unit/test_recommendations.py`
  - [ ] Test pipeline recommender
  - [ ] Test FE recommender
  - [ ] Test HP recommender

---

## Phase 4 : Intégration Pipeline 🟡 IMPORTANT

**Objectif** : Connecter le méta-learning au pipeline existant

### 4.1 Hooks d'intégration
- [ ] **4.1.1** Créer `src/meta_learning/integration/hooks.py`
  - [ ] `on_analysis_complete(project_name, report_path)` - après analyse
    - Extraire méta-features
    - Enregistrer dans MetaDatabase
    - Chercher datasets similaires
    - Retourner recommandations
  - [ ] `on_automl_complete(project_name, framework, score, hp)` - après AutoML
    - Logger résultat dans MetaDatabase
    - Mettre à jour E_matrix

### 4.2 Modification du pipeline
- [ ] **4.2.1** Modifier `src/pipeline/pipeline_all.py`
  - [ ] Ajouter option `use_meta_learning: bool = False`
  - [ ] Appeler hook après phase d'analyse
  - [ ] Passer recommandations aux phases suivantes
  - [ ] Appeler hook après phase AutoML

- [ ] **4.2.2** Modifier `src/automl/runner.py`
  - [ ] Ajouter support warm-start hyperparamètres
  - [ ] Logger résultats automatiquement si meta_learning activé

### 4.3 Enhancement LLMFE (optionnel)
- [ ] **4.3.1** Créer `src/meta_learning/integration/enhance_llmfe.py`
  - [ ] Initialiser LLMFE avec features recommandées
  - [ ] Utiliser stratégies de datasets similaires

### 4.4 Tests d'intégration
- [ ] **4.4.1** Créer `tests/integration/test_meta_learning_pipeline.py`
  - [ ] Test pipeline complet avec meta_learning activé
  - [ ] Vérifier que les hooks sont appelés
  - [ ] Vérifier que les recommandations sont utilisées

---

## Phase 5 : Orchestrateur Unifié 🟢 NICE-TO-HAVE

**Objectif** : API simple pour exécuter le pipeline complet

### 5.1 Structure
- [ ] **5.1.1** Créer la structure
  ```
  src/orchestrator/
  ├── __init__.py
  ├── pipeline.py
  ├── config.py
  └── results.py
  ```

### 5.2 Orchestrateur principal
- [ ] **5.2.1** Créer `src/orchestrator/pipeline.py`
  - [ ] `AutoMLPipeline.__init__(project_name, llm_provider, use_meta_learning)`
  - [ ] `AutoMLPipeline.configure(metric, time_budget, feature_strategies, ...)`
  - [ ] `AutoMLPipeline.run(df_train, df_test, target_col) -> PipelineResult`
  - [ ] `AutoMLPipeline.save(output_dir)`

- [ ] **5.2.2** Créer `src/orchestrator/config.py`
  - [ ] `PipelineConfig` - configuration globale
  - [ ] Validation des paramètres

- [ ] **5.2.3** Créer `src/orchestrator/results.py`
  - [ ] `PipelineResult` - résultats enrichis
  - [ ] best_score, best_model, business_report
  - [ ] similar_datasets, recommended_pipeline

### 5.3 Module validation
- [ ] **5.3.1** Créer `src/validation/metrics.py`
  - [ ] Calcul métriques unifiées

- [ ] **5.3.2** Créer `src/validation/explainability.py`
  - [ ] SHAP values
  - [ ] Feature importance
  - [ ] PDP plots

- [ ] **5.3.3** Créer `src/validation/report_generator.py`
  - [ ] Génération rapport LLM
  - [ ] Export JSON/HTML

### 5.4 Tests
- [ ] **5.4.1** Créer `tests/integration/test_orchestrator.py`
  - [ ] Test pipeline complet via API unifiée

---

## Phase 6 : Fonctionnalités Avancées 🔵 OPTIONNEL

**Objectif** : Fonctionnalités avancées pour améliorer les performances

### 6.1 Landmarking
- [ ] **6.1.1** Créer `src/meta_learning/meta_features/landmarker.py`
  - [ ] `Landmarker.compute(df, target) -> dict`
  - [ ] Quick DecisionTree score (1 fold, max_depth=3)
  - [ ] Quick LogisticRegression score
  - [ ] Quick 1-NN score

### 6.2 Surrogate Model
- [ ] **6.2.1** Créer `src/meta_learning/surrogate/trainer.py`
  - [ ] Entraîner modèle sur E_matrix
  - [ ] Prédire performance pipeline pour nouveau dataset

- [ ] **6.2.2** Créer `src/meta_learning/surrogate/predictor.py`
  - [ ] `SurrogatePredictor.predict(mf_vector, pipeline) -> float`

### 6.3 Transfer Learning
- [ ] **6.3.1** Créer `src/meta_learning/surrogate/transfer_learning.py`
  - [ ] Transférer features entre datasets similaires
  - [ ] Adapter features au nouveau contexte

---

## Checklist Globale

### Documentation
- [ ] README.md pour le module meta_learning
- [ ] Docstrings pour toutes les classes/fonctions publiques
- [ ] Exemples d'utilisation dans docs/

### Tests
- [ ] Coverage > 80% pour nouveau code
- [ ] Tests unitaires pour chaque module
- [ ] Tests d'intégration pour le pipeline complet

### CI/CD
- [ ] Ajouter tests au pipeline CI
- [ ] Vérifier compatibilité avec Python 3.12

### Performance
- [ ] Benchmark LLMFE avec multi-évaluateurs vs mono-évaluateur
- [ ] Benchmark temps de recherche avec/sans méta-learning

---

## Notes d'implémentation

### Dépendances à ajouter
```yaml
# environment.yml
- lightgbm
- catboost  # si pas déjà présent
- optuna    # pour HPO futur
```

### Points d'attention
1. **Rétrocompatibilité** : Le pipeline actuel doit fonctionner sans meta_learning
2. **Performance** : L'extraction de méta-features doit être rapide (< 1s)
3. **Stockage** : JSON pour simplicité, SQLite si volumétrie importante
4. **Concurrence** : Gérer accès concurrent à la base de données

### Ordre d'implémentation recommandé
1. Phase 1.1 (évaluateurs) → testable immédiatement
2. Phase 2.2-2.3 (modèles + extracteur) → base du meta-learning
3. Phase 2.4 (database) → persistance
4. Phase 3.2-3.3 (similarité + recommandations) → coeur du SML-AutoML
5. Phase 4.1-4.2 (hooks + pipeline) → intégration
6. Phases 5-6 → polish et avancé
