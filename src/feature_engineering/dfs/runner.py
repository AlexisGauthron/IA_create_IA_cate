# src/feature_engineering/dfs/runner.py
"""
Runner principal pour Deep Feature Synthesis.

Ce module orchestre le processus complet de DFS :
1. Création de l'EntitySet (avec relations synthétiques optionnelles)
2. Exécution du DFS avec les primitives configurées
3. Sélection des features pertinentes
4. Évaluation multi-modèle
5. Sauvegarde des résultats

AMÉLIORATION : Relations synthétiques
-------------------------------------
Pour les datasets single-table, DFS ne peut pas utiliser les primitives
d'agrégation (max_depth effectif = 1). La solution : créer des "pseudo-tables"
en groupant par colonnes catégorielles.

Exemple pour Titanic :
- Table "passengers" (principale)
- Table "pclass_stats" : stats par classe
- Table "embarked_stats" : stats par port d'embarquement
- Table "sex_stats" : stats par sexe

Cela permet des features comme "MEAN(pclass_stats.Age)" ou
"MAX(embarked_stats.Fare) - Fare" (écart au max du groupe).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    import featuretools as ft
    from woodwork.logical_types import (
        Boolean,
        Categorical,
        Datetime,
        Double,
        Integer,
    )

    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False

from src.feature_engineering.dfs.config import DFSConfig
from src.feature_engineering.dfs.primitives import get_primitives_for_task
from src.feature_engineering.dfs.selection import FeatureSelector, SelectionResult
from src.feature_engineering.path_config import FeatureEngineeringPathConfig


@dataclass
class DFSResult:
    """Résultat complet d'un run DFS."""

    # Identifiants
    project_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Features générées
    n_features_generated: int = 0
    n_features_selected: int = 0
    feature_names: list[str] = field(default_factory=list)
    feature_definitions: list[str] = field(default_factory=list)

    # Sélection
    selection_result: SelectionResult | None = None

    # Scores
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement_pct: float = 0.0
    scores_by_model: dict[str, float] = field(default_factory=dict)

    # Temps
    execution_time_seconds: float = 0.0
    dfs_time_seconds: float = 0.0
    selection_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        data = asdict(self)
        # SelectionResult n'est pas directement sérialisable
        if self.selection_result:
            data["selection_result"] = asdict(self.selection_result)
        return data


class DFSRunner:
    """
    Runner pour exécuter Deep Feature Synthesis sur un dataset.

    Exemple d'utilisation:
    ```python
    runner = DFSRunner(project_name="titanic")
    result = runner.run(
        df_train=df,
        target_col="Survived",
        is_regression=False,
    )
    ```
    """

    def __init__(
        self,
        project_name: str,
        config: DFSConfig | None = None,
        path_config: FeatureEngineeringPathConfig | None = None,
    ):
        """
        Initialise le runner DFS.

        Args:
            project_name: Nom du projet
            config: Configuration DFS (utilise les defaults si None)
            path_config: Configuration des chemins (créé automatiquement si None)
        """
        if not FEATURETOOLS_AVAILABLE:
            raise ImportError(
                "featuretools n'est pas installé. " "Installez-le avec: pip install featuretools"
            )

        self.project_name = project_name
        self.config = config or DFSConfig()
        self.path_config = path_config

        self._es: ft.EntitySet | None = None
        self._feature_defs: list = []
        self._feature_matrix: pd.DataFrame | None = None

    def run(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        df_test: pd.DataFrame | None = None,
        is_regression: bool = False,
        index_col: str | None = None,
        time_index: str | None = None,
        datetime_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        ignore_cols: list[str] | None = None,
    ) -> DFSResult:
        """
        Exécute DFS sur le dataset.

        Args:
            df_train: DataFrame d'entraînement
            target_col: Colonne cible
            df_test: DataFrame de test (optionnel)
            is_regression: True si régression
            index_col: Colonne d'index (créé automatiquement si None)
            time_index: Colonne temporelle pour le tri
            datetime_cols: Colonnes de type datetime
            categorical_cols: Colonnes catégorielles
            ignore_cols: Colonnes à ignorer dans DFS

        Returns:
            DFSResult avec toutes les métriques
        """
        start_time = time.time()

        # Initialiser path_config si nécessaire
        if self.path_config is None:
            self.path_config = FeatureEngineeringPathConfig(project_name=self.project_name)

        # Créer le dossier DFS
        dfs_dir = self.path_config.project_dir / "dfs"
        dfs_dir.mkdir(parents=True, exist_ok=True)

        if self.config.verbose:
            self._print_header(df_train, target_col, is_regression)

        # Préparer les données
        df_work = df_train.copy()
        y = df_work.pop(target_col)

        # Ajouter un index si nécessaire
        if index_col is None:
            index_col = "_dfs_index"
            df_work[index_col] = range(len(df_work))

        # Colonnes à ignorer
        ignore_cols = ignore_cols or []
        if index_col not in ignore_cols:
            ignore_cols_with_index = ignore_cols + [index_col]
        else:
            ignore_cols_with_index = ignore_cols

        # Détecter les types de colonnes
        has_datetime = datetime_cols is not None and len(datetime_cols) > 0
        has_text = any(df_work[col].dtype == "object" for col in df_work.columns)

        # Récupérer les primitives selon la configuration
        agg_primitives, trans_primitives = self._get_primitives(
            has_datetime=has_datetime,
            has_text=has_text,
        )

        # === ÉTAPE 1: Créer l'EntitySet ===
        if self.config.verbose:
            print("\n[DFS] Étape 1: Création de l'EntitySet...")

        self._es = self._create_entityset(
            df=df_work,
            index_col=index_col,
            time_index=time_index,
            datetime_cols=datetime_cols,
            categorical_cols=categorical_cols,
        )

        # === ÉTAPE 2: Exécuter DFS ===
        if self.config.verbose:
            print(f"\n[DFS] Étape 2: Exécution de DFS (max_depth={self.config.max_depth})...")
            print(f"      Aggregation primitives: {agg_primitives}")
            print(f"      Transform primitives: {trans_primitives}")

        dfs_start = time.time()

        # Préparer les kwargs pour ft.dfs
        dfs_kwargs = {
            "entityset": self._es,
            "target_dataframe_name": "main",
            "agg_primitives": agg_primitives,
            "trans_primitives": trans_primitives,
            "max_depth": self.config.max_depth,
            "ignore_columns": {"main": ignore_cols_with_index},
            "verbose": self.config.verbose,
        }

        # Ajouter max_features seulement si spécifié (None cause une erreur)
        if self.config.max_features is not None:
            dfs_kwargs["max_features"] = self.config.max_features

        # n_jobs et chunk_size peuvent causer des problèmes
        if self.config.n_jobs != -1:
            dfs_kwargs["n_jobs"] = self.config.n_jobs
        if self.config.chunk_size is not None:
            dfs_kwargs["chunk_size"] = self.config.chunk_size

        self._feature_matrix, self._feature_defs = ft.dfs(**dfs_kwargs)

        dfs_time = time.time() - dfs_start

        n_features_generated = self._feature_matrix.shape[1]
        if self.config.verbose:
            print(f"[DFS] Features générées: {n_features_generated}")

        # Sauvegarder les définitions des features
        feature_definitions = [str(f) for f in self._feature_defs]

        # === ÉTAPE 3: Sélection des features ===
        selection_result = None
        selection_time = 0.0

        if self.config.feature_selection and n_features_generated > 0:
            if self.config.verbose:
                print(
                    f"\n[DFS] Étape 3: Sélection des features ({self.config.selection_method})..."
                )

            selection_start = time.time()

            selector = FeatureSelector(
                method=self.config.selection_method,
                threshold=self.config.selection_threshold,
                correlation_threshold=self.config.correlation_threshold,
                top_k=self.config.top_k_features,
                is_regression=is_regression,
                random_state=self.config.random_state,
                verbose=self.config.verbose,
            )

            self._feature_matrix, selection_result = selector.fit_transform(self._feature_matrix, y)

            selection_time = time.time() - selection_start

        n_features_selected = self._feature_matrix.shape[1]

        # === ÉTAPE 4: Évaluation ===
        if self.config.verbose:
            print("\n[DFS] Étape 4: Évaluation des features...")

        eval_start = time.time()

        initial_score, scores_by_model = self._evaluate_features(
            X_original=df_train.drop(columns=[target_col]),
            X_dfs=self._feature_matrix,
            y=y,
            is_regression=is_regression,
        )

        final_score = np.mean(list(scores_by_model.values()))
        eval_time = time.time() - eval_start

        # Calculer l'amélioration
        if initial_score > 0:
            improvement_pct = ((final_score - initial_score) / initial_score) * 100
        else:
            improvement_pct = 0.0

        # === ÉTAPE 5: Sauvegarde ===
        total_time = time.time() - start_time

        result = DFSResult(
            project_name=self.project_name,
            n_features_generated=n_features_generated,
            n_features_selected=n_features_selected,
            feature_names=list(self._feature_matrix.columns),
            feature_definitions=feature_definitions[:n_features_selected],
            selection_result=selection_result,
            initial_score=initial_score,
            final_score=final_score,
            improvement_pct=improvement_pct,
            scores_by_model=scores_by_model,
            execution_time_seconds=total_time,
            dfs_time_seconds=dfs_time,
            selection_time_seconds=selection_time,
            evaluation_time_seconds=eval_time,
            config=self.config.to_dict(),
        )

        # Sauvegarder les résultats
        self._save_results(result, dfs_dir)

        # Sauvegarder le DataFrame transformé
        output_path = dfs_dir / "train_dfs.parquet"
        self._feature_matrix[target_col] = y.values
        self._feature_matrix.to_parquet(output_path)

        if self.config.verbose:
            self._print_summary(result)

        return result

    def _get_primitives(
        self,
        has_datetime: bool,
        has_text: bool,
    ) -> tuple[list[str], list[str]]:
        """Récupère les primitives à utiliser."""
        if self.config.agg_primitives and self.config.trans_primitives:
            return self.config.agg_primitives, self.config.trans_primitives

        # Déterminer le niveau de complexité
        complexity = min(self.config.max_depth, 3)

        agg_prims, trans_prims = get_primitives_for_task(
            has_datetime=has_datetime,
            has_text=has_text,
            complexity_level=complexity,
        )

        # Utiliser les primitives configurées si spécifiées
        if self.config.agg_primitives:
            agg_prims = self.config.agg_primitives
        if self.config.trans_primitives:
            trans_prims = self.config.trans_primitives

        return agg_prims, trans_prims

    def _create_entityset(
        self,
        df: pd.DataFrame,
        index_col: str,
        time_index: str | None = None,
        datetime_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> ft.EntitySet:
        """Crée l'EntitySet pour Featuretools."""
        es = ft.EntitySet(id=f"{self.project_name}_es")

        # Définir les types logiques
        logical_types = {}

        if datetime_cols:
            for col in datetime_cols:
                if col in df.columns:
                    logical_types[col] = Datetime

        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    logical_types[col] = Categorical

        # Détecter automatiquement les types
        for col in df.columns:
            if col in logical_types or col == index_col:
                continue

            dtype = df[col].dtype

            if dtype == "bool":
                logical_types[col] = Boolean
            elif dtype in ["int64", "int32"]:
                logical_types[col] = Integer
            elif dtype in ["float64", "float32"]:
                logical_types[col] = Double
            elif dtype == "object":
                # Vérifier si c'est une date
                try:
                    pd.to_datetime(df[col].dropna().iloc[:100])
                    logical_types[col] = Datetime
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    # Vérifier la cardinalité
                    if df[col].nunique() < 50:
                        logical_types[col] = Categorical

        # === Préparer les relations synthétiques AVANT d'ajouter à l'EntitySet ===
        synthetic_tables = {}
        if self.config.create_synthetic_relations:
            df, synthetic_tables = self._prepare_synthetic_relations(
                df=df,
                index_col=index_col,
            )
            # Ajouter les types pour les colonnes de jointure
            for col_name in df.columns:
                if col_name.startswith("_") and col_name.endswith("_group_id"):
                    logical_types[col_name] = Integer

        es = es.add_dataframe(
            dataframe_name="main",
            dataframe=df,
            index=index_col,
            time_index=time_index,
            logical_types=logical_types if logical_types else None,
        )

        # === Ajouter les tables synthétiques et créer les relations ===
        if synthetic_tables:
            es = self._add_synthetic_tables(es, synthetic_tables)

        return es

    def _prepare_synthetic_relations(
        self,
        df: pd.DataFrame,
        index_col: str,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Prépare les colonnes de jointure et les tables de groupes.

        Returns:
            Tuple (df avec colonnes de jointure, dict des tables de groupes)
        """
        if self.config.verbose:
            print("\n[DFS] Préparation des relations synthétiques...")

        # Déterminer les colonnes à utiliser pour les groupes
        if self.config.synthetic_groupby_cols:
            groupby_cols = [col for col in self.config.synthetic_groupby_cols if col in df.columns]
        else:
            groupby_cols = self._detect_groupby_columns(df, index_col)

        groupby_cols = groupby_cols[: self.config.max_synthetic_tables]

        if self.config.verbose:
            print(f"      Colonnes pour groupes: {groupby_cols}")

        if not groupby_cols:
            return df, {}

        synthetic_tables = {}
        df = df.copy()

        for col in groupby_cols:
            # Vérifier la taille minimale des groupes
            group_sizes = df[col].value_counts()
            valid_groups = group_sizes[group_sizes >= self.config.min_group_size]

            if len(valid_groups) < 2:
                if self.config.verbose:
                    print(f"      Skipping {col}: pas assez de groupes valides")
                continue

            # Mapper les valeurs vers des IDs numériques
            unique_values = df[col].dropna().unique()
            value_to_id = {v: i for i, v in enumerate(unique_values)}

            # Créer la colonne de jointure (utiliser pd.Series pour éviter les problèmes de type)
            join_col_name = f"_{col}_group_id"
            mapped_values = df[col].map(value_to_id)
            # Convertir en float puis en int pour gérer les NaN
            df[join_col_name] = mapped_values.where(mapped_values.notna(), -1).astype(int)

            # Créer la table de groupes
            group_df = pd.DataFrame(
                {
                    f"{col}_id": list(range(len(unique_values))),
                    col: list(unique_values),
                }
            )

            table_name = f"{col}_groups"
            synthetic_tables[table_name] = {
                "df": group_df,
                "index": f"{col}_id",
                "join_col": join_col_name,
            }

            if self.config.verbose:
                print(f"      ✓ Préparé '{table_name}' ({len(group_df)} groupes)")

        return df, synthetic_tables

    def _add_synthetic_tables(
        self,
        es: ft.EntitySet,
        synthetic_tables: dict,
    ) -> ft.EntitySet:
        """Ajoute les tables synthétiques et crée les relations."""
        for table_name, info in synthetic_tables.items():
            try:
                es = es.add_dataframe(
                    dataframe_name=table_name,
                    dataframe=info["df"],
                    index=info["index"],
                )

                es = es.add_relationship(
                    parent_dataframe_name=table_name,
                    parent_column_name=info["index"],
                    child_dataframe_name="main",
                    child_column_name=info["join_col"],
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"      ✗ Erreur pour {table_name}: {e}")

        return es

    def _detect_groupby_columns(
        self,
        df: pd.DataFrame,
        index_col: str,
    ) -> list[str]:
        """
        Détecte automatiquement les colonnes appropriées pour créer des groupes.

        Critères :
        - Type object ou category ou int avec peu de valeurs uniques
        - Cardinalité entre 2 et 50
        - Au moins min_group_size éléments par groupe en moyenne
        """
        groupby_candidates = []

        for col in df.columns:
            if col == index_col:
                continue

            dtype = df[col].dtype
            nunique = df[col].nunique()

            # Critères de sélection
            is_categorical = dtype in ["object", "category"]
            is_low_cardinality_int = dtype in ["int64", "int32"] and nunique <= 20

            if (is_categorical or is_low_cardinality_int) and 2 <= nunique <= 50:
                # Vérifier que les groupes ont assez d'éléments
                avg_group_size = len(df) / nunique
                if avg_group_size >= self.config.min_group_size:
                    groupby_candidates.append((col, nunique, avg_group_size))

        # Trier par taille moyenne des groupes (préférer les groupes plus grands)
        groupby_candidates.sort(key=lambda x: x[2], reverse=True)

        return [col for col, _, _ in groupby_candidates]

    def _evaluate_features(
        self,
        X_original: pd.DataFrame,
        X_dfs: pd.DataFrame,
        y: np.ndarray | pd.Series,
        is_regression: bool,
    ) -> tuple[float, dict[str, float]]:
        """Évalue les features avec plusieurs modèles."""
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        # Encoder la target si nécessaire
        y_encoded = y
        if not is_regression and hasattr(y, "dtype") and y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        # Préparer X_original
        X_orig_clean = self._prepare_for_eval(X_original)

        # Préparer X_dfs
        X_dfs_clean = self._prepare_for_eval(X_dfs)

        # Définir la métrique
        if self.config.eval_metric == "auto":
            scoring = "neg_root_mean_squared_error" if is_regression else "f1_weighted"
        else:
            scoring = self.config.eval_metric

        scores_by_model = {}

        for model_name in self.config.eval_models:
            model = self._get_model(model_name, is_regression)
            if model is None:
                continue

            try:
                # Score avec features DFS
                dfs_scores = cross_val_score(
                    model,
                    X_dfs_clean,
                    y_encoded,
                    cv=self.config.cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                )

                # Pour RMSE, on a des valeurs négatives
                if "neg_" in scoring:
                    scores_by_model[model_name] = -np.mean(dfs_scores)
                else:
                    scores_by_model[model_name] = np.mean(dfs_scores)

                if self.config.verbose:
                    print(f"      {model_name}: {scores_by_model[model_name]:.4f}")

            except Exception as e:
                if self.config.verbose:
                    print(f"      {model_name}: Erreur - {e}")

        # Calculer le score initial (sans DFS)
        try:
            base_model = self._get_model("xgboost", is_regression)
            if base_model:
                orig_scores = cross_val_score(
                    base_model,
                    X_orig_clean,
                    y_encoded,
                    cv=self.config.cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                )
                if "neg_" in scoring:
                    initial_score = -np.mean(orig_scores)
                else:
                    initial_score = np.mean(orig_scores)
            else:
                initial_score = 0.0
        except Exception:
            initial_score = 0.0

        return initial_score, scores_by_model

    def _prepare_for_eval(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prépare un DataFrame pour l'évaluation."""
        X_clean = X.copy()

        # Supprimer colonnes avec trop de NaN
        nan_ratio = X_clean.isna().mean()
        X_clean = X_clean.loc[:, nan_ratio < 0.5]

        # Remplir les NaN
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                if X_clean[col].dtype.name == "category":
                    # Pour les colonnes category, ajouter d'abord la catégorie
                    if "MISSING" not in X_clean[col].cat.categories:
                        X_clean[col] = X_clean[col].cat.add_categories(["MISSING"])
                    X_clean[col] = X_clean[col].fillna("MISSING")
                elif X_clean[col].dtype == "object":
                    X_clean[col] = X_clean[col].fillna("MISSING")
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())

        # Encoder les catégorielles
        for col in X_clean.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))

        # Supprimer colonnes infinies
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(0)

        return X_clean

    def _get_model(self, model_name: str, is_regression: bool):
        """Retourne une instance du modèle demandé."""
        try:
            if model_name == "xgboost":
                from xgboost import XGBClassifier, XGBRegressor

                if is_regression:
                    return XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        verbosity=0,
                    )
                else:
                    return XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        verbosity=0,
                        use_label_encoder=False,
                        eval_metric="logloss",
                    )

            elif model_name == "lightgbm":
                from lightgbm import LGBMClassifier, LGBMRegressor

                if is_regression:
                    return LGBMRegressor(
                        n_estimators=100,
                        max_depth=6,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        verbose=-1,
                    )
                else:
                    return LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        verbose=-1,
                    )

            elif model_name == "randomforest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

                if is_regression:
                    return RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    )
                else:
                    return RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    )

        except ImportError:
            return None

        return None

    def _save_results(self, result: DFSResult, output_dir: Path) -> None:
        """Sauvegarde les résultats."""
        # Sauvegarder le rapport JSON
        report_path = output_dir / "dfs_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        # Sauvegarder les définitions des features
        features_path = output_dir / "feature_definitions.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "features": result.feature_names,
                    "definitions": result.feature_definitions,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if self.config.verbose:
            print(f"\n[DFS] Résultats sauvegardés dans: {output_dir}")

    def _print_header(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_regression: bool,
    ) -> None:
        """Affiche l'en-tête."""
        print("\n" + "=" * 70)
        print("           DEEP FEATURE SYNTHESIS (DFS)")
        print("=" * 70)
        print(f"  Projet      : {self.project_name}")
        print(f"  Dataset     : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"  Target      : {target_col}")
        print(f"  Type        : {'Régression' if is_regression else 'Classification'}")
        print(f"  Max depth   : {self.config.max_depth}")
        print(
            f"  Sélection   : {self.config.selection_method if self.config.feature_selection else 'Désactivée'}"
        )
        print("=" * 70)

    def _print_summary(self, result: DFSResult) -> None:
        """Affiche le résumé."""
        print("\n" + "─" * 70)
        print("  RÉSUMÉ DFS")
        print("─" * 70)
        print(f"  Features générées   : {result.n_features_generated}")
        print(f"  Features sélectionnées : {result.n_features_selected}")
        print(f"  Score initial       : {result.initial_score:.4f}")
        print(f"  Score final         : {result.final_score:.4f}")
        print(f"  Amélioration        : {result.improvement_pct:+.2f}%")
        print(f"  Temps total         : {result.execution_time_seconds:.1f}s")
        print("─" * 70)

    def get_feature_matrix(self) -> pd.DataFrame | None:
        """Retourne la matrice de features générée."""
        return self._feature_matrix

    def get_feature_definitions(self) -> list:
        """Retourne les définitions des features."""
        return self._feature_defs


def run_dfs(
    df_train: pd.DataFrame,
    target_col: str,
    project_name: str,
    df_test: pd.DataFrame | None = None,
    is_regression: bool = False,
    max_depth: int = 2,
    feature_selection: bool = True,
    selection_method: str = "hybrid",
    top_k_features: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, DFSResult]:
    """
    Fonction raccourcie pour exécuter DFS.

    Args:
        df_train: DataFrame d'entraînement
        target_col: Colonne cible
        project_name: Nom du projet
        df_test: DataFrame de test (optionnel)
        is_regression: Type de tâche
        max_depth: Profondeur maximale DFS
        feature_selection: Activer la sélection
        selection_method: Méthode de sélection
        top_k_features: Nombre max de features
        verbose: Afficher les logs

    Returns:
        Tuple (DataFrame transformé, DFSResult)

    Example:
        >>> df_transformed, result = run_dfs(
        ...     df_train=train_df,
        ...     target_col="Survived",
        ...     project_name="titanic_dfs",
        ...     max_depth=2,
        ... )
    """
    config = DFSConfig(
        max_depth=max_depth,
        feature_selection=feature_selection,
        selection_method=selection_method,
        top_k_features=top_k_features,
        verbose=verbose,
    )

    runner = DFSRunner(project_name=project_name, config=config)
    result = runner.run(
        df_train=df_train,
        target_col=target_col,
        df_test=df_test,
        is_regression=is_regression,
    )

    return runner.get_feature_matrix(), result
