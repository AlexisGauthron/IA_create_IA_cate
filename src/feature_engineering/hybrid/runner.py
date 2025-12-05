# src/feature_engineering/hybrid/runner.py
"""
Runner principal pour le Feature Engineering Hybride (LLMFE + DFS).

Pipeline hybride :
1. LLMFE génère des features métier (connaissance business via LLM)
2. DFS enrichit avec des agrégations et interactions structurelles
3. Sélection finale pour éliminer la redondance

Priorité : Les features LLMFE sont prioritaires car plus interprétables
et alignées avec la logique métier.
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering.hybrid.config import HybridConfig
from src.feature_engineering.llmfe.feature_formatter import FeatureFormat
from src.feature_engineering.path_config import FeatureEngineeringPathConfig


@dataclass
class HybridResult:
    """Résultat complet du Feature Engineering Hybride."""

    # Identifiants
    project_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Features
    original_features: list[str] = field(default_factory=list)
    llmfe_features: list[str] = field(default_factory=list)
    dfs_features: list[str] = field(default_factory=list)
    final_features: list[str] = field(default_factory=list)

    # Compteurs
    n_original: int = 0
    n_llmfe_added: int = 0
    n_dfs_added: int = 0
    n_final: int = 0
    n_dropped: int = 0

    # Scores
    baseline_score: float = 0.0
    llmfe_score: float = 0.0
    dfs_score: float = 0.0
    final_score: float = 0.0
    improvement_pct: float = 0.0

    # Temps
    total_time_seconds: float = 0.0
    llmfe_time_seconds: float = 0.0
    dfs_time_seconds: float = 0.0
    selection_time_seconds: float = 0.0

    # Détails
    llmfe_result: Any = None
    dfs_result: Any = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        data = asdict(self)
        # Nettoyer les objets non sérialisables
        data["llmfe_result"] = str(self.llmfe_result) if self.llmfe_result else None
        data["dfs_result"] = str(self.dfs_result) if self.dfs_result else None
        return data


class HybridFeatureEngineer:
    """
    Feature Engineering Hybride combinant LLMFE et DFS.

    Flux de données :
    1. Données brutes → LLMFE → Features métier (family_size, title, etc.)
    2. Données + LLMFE features → DFS → Features structurelles (agrégations)
    3. Toutes les features → Sélection finale → Features optimales

    Priorité LLMFE : En cas de features similaires, LLMFE est prioritaire
    car les features métier sont plus interprétables et pertinentes.

    Exemple d'utilisation:
    ```python
    from src.feature_engineering.hybrid import HybridFeatureEngineer, HybridConfig

    config = HybridConfig(
        llmfe_max_iterations=10,
        dfs_config="synthetic_standard",
        max_features=50,
    )

    engineer = HybridFeatureEngineer(project_name="titanic", config=config)
    result = engineer.run(
        df_train=train_df,
        target_col="Survived",
        is_regression=False,
    )

    # Accéder aux données transformées
    df_transformed = engineer.get_transformed_data()
    ```
    """

    def __init__(
        self,
        project_name: str,
        config: HybridConfig | None = None,
        path_config: FeatureEngineeringPathConfig | None = None,
    ):
        """
        Initialise le HybridFeatureEngineer.

        Args:
            project_name: Nom du projet
            config: Configuration hybride (défaut si None)
            path_config: Configuration des chemins
        """
        self.project_name = project_name
        self.config = config or HybridConfig()
        self.path_config = path_config

        # Runners (initialisés lors du run)
        self._llmfe_runner = None
        self._dfs_runner = None

        # Résultats
        self._df_transformed: pd.DataFrame | None = None
        self._result: HybridResult | None = None

    def run(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        df_test: pd.DataFrame | None = None,
        is_regression: bool = False,
        analyse_path: Path | str | None = None,
        task_description: str | None = None,
    ) -> HybridResult:
        """
        Exécute le Feature Engineering Hybride.

        Args:
            df_train: DataFrame d'entraînement
            target_col: Colonne cible
            df_test: DataFrame de test (optionnel)
            is_regression: True si régression
            analyse_path: Chemin vers le rapport d'analyse (pour LLMFE)
            task_description: Description de la tâche (pour LLMFE)

        Returns:
            HybridResult avec toutes les métriques
        """
        start_time = time.time()

        # Initialiser les chemins
        if self.path_config is None:
            self.path_config = FeatureEngineeringPathConfig(project_name=self.project_name)

        # Créer le dossier hybrid
        hybrid_dir = self.path_config.project_dir / "hybrid"
        hybrid_dir.mkdir(parents=True, exist_ok=True)

        if self.config.verbose:
            self._print_header(df_train, target_col, is_regression)

        # Sauvegarder les features originales
        original_features = [c for c in df_train.columns if c != target_col]
        y = df_train[target_col]

        # Calculer le score baseline
        baseline_score = self._calculate_baseline_score(
            df_train.drop(columns=[target_col]), y, is_regression
        )

        if self.config.verbose:
            print(f"\n[BASELINE] Score initial: {baseline_score:.4f}")

        # Variables pour tracker les résultats
        df_current = df_train.copy()
        llmfe_features = []
        dfs_features = []
        llmfe_result = None
        dfs_result = None
        llmfe_time = 0.0
        dfs_time = 0.0
        llmfe_score = baseline_score
        dfs_score = baseline_score

        # === ÉTAPE 1: LLMFE (Features métier) ===
        if self.config.enable_llmfe:
            if self.config.verbose:
                print("\n" + "=" * 60)
                print("  ÉTAPE 1: LLMFE (Features métier)")
                print("=" * 60)

            llmfe_start = time.time()

            try:
                df_current, llmfe_result, llmfe_features = self._run_llmfe(
                    df_train=df_current,
                    target_col=target_col,
                    is_regression=is_regression,
                    analyse_path=analyse_path,
                    task_description=task_description,
                )

                llmfe_time = time.time() - llmfe_start

                # Calculer le score après LLMFE
                llmfe_score = self._calculate_score(
                    df_current.drop(columns=[target_col]), y, is_regression
                )

                if self.config.verbose:
                    print(f"\n[LLMFE] Features ajoutées: {len(llmfe_features)}")
                    print(f"[LLMFE] Score: {llmfe_score:.4f} ({llmfe_score - baseline_score:+.4f})")
                    print(f"[LLMFE] Temps: {llmfe_time:.1f}s")

            except Exception as e:
                if self.config.verbose:
                    print(f"\n[LLMFE] Erreur: {e}")

                # Tenter de récupérer le fichier même si une erreur s'est produite
                # (le fichier peut avoir été sauvegardé avant l'erreur)
                llmfe_path_config = FeatureEngineeringPathConfig(
                    project_name=f"{self.project_name}_llmfe"
                )
                train_fe_csv_path = llmfe_path_config.train_fe_csv_path

                if train_fe_csv_path.exists():
                    try:
                        df_current = pd.read_csv(train_fe_csv_path)
                        if target_col not in df_current.columns:
                            df_current[target_col] = df_train[target_col].values
                        features_after = (
                            set(df_current.columns) - set(df_train.columns) - {target_col}
                        )
                        llmfe_features = list(features_after)

                        # Récupérer le score depuis summary.json
                        summary_path = llmfe_path_config.llmfe_results_dir / "summary.json"
                        if summary_path.exists():
                            with open(summary_path) as f:
                                summary = json.load(f)
                                llmfe_score = summary.get("best_score", baseline_score)

                        if self.config.verbose:
                            print(
                                f"[LLMFE] Récupéré {len(llmfe_features)} features depuis le fichier sauvegardé"
                            )
                            print(f"[LLMFE] Score récupéré: {llmfe_score:.4f}")
                            if llmfe_features:
                                print(
                                    f"[LLMFE] Features: {llmfe_features[:5]}{'...' if len(llmfe_features) > 5 else ''}"
                                )
                    except Exception as e2:
                        if self.config.verbose:
                            print(f"[LLMFE] Impossible de récupérer le fichier: {e2}")
                            print("[LLMFE] Continuation avec DFS seul...")
                else:
                    if self.config.verbose:
                        print("[LLMFE] Continuation avec DFS seul...")

                llmfe_time = time.time() - llmfe_start

        # === ÉTAPE 2: DFS (Features structurelles) ===
        if self.config.enable_dfs:
            if self.config.verbose:
                print("\n" + "=" * 60)
                print("  ÉTAPE 2: DFS (Features structurelles)")
                print("=" * 60)

            dfs_start = time.time()

            try:
                df_current, dfs_result, dfs_features = self._run_dfs(
                    df_train=df_current,
                    target_col=target_col,
                    is_regression=is_regression,
                )

                dfs_time = time.time() - dfs_start

                # Calculer le score après DFS
                dfs_score = self._calculate_score(
                    df_current.drop(columns=[target_col]), y, is_regression
                )

                if self.config.verbose:
                    print(f"\n[DFS] Features ajoutées: {len(dfs_features)}")
                    print(f"[DFS] Score: {dfs_score:.4f} ({dfs_score - llmfe_score:+.4f})")
                    print(f"[DFS] Temps: {dfs_time:.1f}s")

            except Exception as e:
                if self.config.verbose:
                    print(f"\n[DFS] Erreur: {e}")
                dfs_time = time.time() - dfs_start

        # === ÉTAPE 3: Sélection finale ===
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("  ÉTAPE 3: Sélection finale")
            print("=" * 60)

        selection_start = time.time()

        df_final, final_features, feature_importances = self._final_selection(
            df=df_current,
            target_col=target_col,
            original_features=original_features,
            llmfe_features=llmfe_features,
            dfs_features=dfs_features,
            is_regression=is_regression,
        )

        selection_time = time.time() - selection_start

        # Calculer le score final
        final_score = self._calculate_score(df_final.drop(columns=[target_col]), y, is_regression)

        # Calculer l'amélioration
        if baseline_score > 0:
            improvement_pct = ((final_score - baseline_score) / baseline_score) * 100
        else:
            improvement_pct = 0.0

        total_time = time.time() - start_time

        # Créer le résultat
        self._result = HybridResult(
            project_name=self.project_name,
            original_features=original_features,
            llmfe_features=llmfe_features,
            dfs_features=dfs_features,
            final_features=final_features,
            n_original=len(original_features),
            n_llmfe_added=len(llmfe_features),
            n_dfs_added=len(dfs_features),
            n_final=len(final_features),
            n_dropped=len(original_features)
            + len(llmfe_features)
            + len(dfs_features)
            - len(final_features),
            baseline_score=baseline_score,
            llmfe_score=llmfe_score,
            dfs_score=dfs_score,
            final_score=final_score,
            improvement_pct=improvement_pct,
            total_time_seconds=total_time,
            llmfe_time_seconds=llmfe_time,
            dfs_time_seconds=dfs_time,
            selection_time_seconds=selection_time,
            llmfe_result=llmfe_result,
            dfs_result=dfs_result,
            feature_importances=feature_importances,
            config=self.config.to_dict(),
        )

        # Sauvegarder les résultats
        self._df_transformed = df_final
        self._save_results(hybrid_dir)

        if self.config.verbose:
            self._print_summary()

        return self._result

    def _run_llmfe(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        is_regression: bool,
        analyse_path: Path | str | None = None,
        task_description: str | None = None,
    ) -> tuple[pd.DataFrame, Any, list[str]]:
        """Exécute LLMFE et retourne le DataFrame enrichi."""
        from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner

        # Créer le path config pour LLMFE
        llmfe_path_config = FeatureEngineeringPathConfig(project_name=f"{self.project_name}_llmfe")

        self._llmfe_runner = LLMFERunner(
            project_name=self.project_name,
            path_config=llmfe_path_config,
        )

        # Features avant LLMFE
        features_before = set(df_train.columns)

        # Convertir le format string en Enum FeatureFormat
        feature_format = self.config.llmfe_feature_format
        if isinstance(feature_format, str):
            feature_format = FeatureFormat(feature_format.lower())

        # Exécuter LLMFE
        result = self._llmfe_runner.run(
            df_train=df_train,
            target_col=target_col,
            is_regression=is_regression,
            max_samples=self.config.llmfe_max_iterations,
            feature_format=feature_format,
            eval_models=self.config.llmfe_eval_models,
            analyse_path=analyse_path,
            task_description=task_description,
        )

        # Charger le DataFrame transformé
        # LLMFE sauvegarde en CSV dans dataset_fe/ (pas en parquet dans llmfe/)
        train_fe_csv_path = llmfe_path_config.train_fe_csv_path
        train_fe_parquet_path = llmfe_path_config.train_features_path

        df_transformed = None

        # 1. Essayer le CSV (sauvegardé par LLMFE profile.py)
        if train_fe_csv_path.exists():
            df_transformed = pd.read_csv(train_fe_csv_path)
            if self.config.verbose:
                print(f"[LLMFE] Chargé depuis: {train_fe_csv_path}")

        # 2. Essayer le Parquet (au cas où)
        elif train_fe_parquet_path.exists():
            df_transformed = pd.read_parquet(train_fe_parquet_path)
            if self.config.verbose:
                print(f"[LLMFE] Chargé depuis: {train_fe_parquet_path}")

        # 3. Fallback : utiliser le DataFrame original
        if df_transformed is None:
            if self.config.verbose:
                print("[LLMFE] ⚠️ Fichier transformé non trouvé, utilisation des données originales")
                print("        Chemins vérifiés:")
                print(f"        - {train_fe_csv_path}")
                print(f"        - {train_fe_parquet_path}")
            df_transformed = df_train

        # S'assurer que la target est présente
        if target_col not in df_transformed.columns:
            df_transformed[target_col] = df_train[target_col].values
            if self.config.verbose:
                print(f"[LLMFE] Target '{target_col}' ajoutée au DataFrame")

        # Identifier les nouvelles features
        features_after = set(df_transformed.columns)
        new_features = list(features_after - features_before - {target_col})

        if self.config.verbose and new_features:
            print(
                f"[LLMFE] Nouvelles features détectées: {new_features[:5]}{'...' if len(new_features) > 5 else ''}"
            )

        return df_transformed, result, new_features

    def _run_dfs(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        is_regression: bool,
    ) -> tuple[pd.DataFrame, Any, list[str]]:
        """Exécute DFS et retourne le DataFrame enrichi."""
        from src.feature_engineering.dfs import DFSRunner

        # Utiliser la config DFS
        dfs_config = self.config.dfs_config

        # Créer le runner DFS
        self._dfs_runner = DFSRunner(
            project_name=f"{self.project_name}_dfs",
            config=dfs_config,
        )

        # Features avant DFS
        features_before = set(df_train.columns)

        # Exécuter DFS
        result = self._dfs_runner.run(
            df_train=df_train,
            target_col=target_col,
            is_regression=is_regression,
        )

        # Récupérer le DataFrame transformé
        df_transformed = self._dfs_runner.get_feature_matrix()

        if df_transformed is None:
            return df_train, result, []

        # Ajouter la target si absente
        if target_col not in df_transformed.columns:
            df_transformed[target_col] = df_train[target_col].values

        # Identifier les nouvelles features
        features_after = set(df_transformed.columns)
        new_features = list(features_after - features_before - {target_col})

        return df_transformed, result, new_features

    def _final_selection(
        self,
        df: pd.DataFrame,
        target_col: str,
        original_features: list[str],
        llmfe_features: list[str],
        dfs_features: list[str],
        is_regression: bool,
    ) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
        """
        Sélection finale des features avec priorité LLMFE.

        Stratégie :
        1. Supprimer les features constantes et avec trop de NaN
        2. Supprimer les features trop corrélées (garder LLMFE en priorité)
        3. Calculer l'importance et filtrer
        4. Appliquer max_features si spécifié
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if self.config.verbose:
            print(f"[SELECTION] Features avant sélection: {X.shape[1]}")

        # Préparer pour l'évaluation
        X_clean = self._prepare_for_eval(X)

        # Encoder la target si nécessaire
        y_encoded = y
        if not is_regression and y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        # 1. Supprimer features constantes
        nunique = X_clean.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        X_clean = X_clean.drop(columns=constant_cols)

        if self.config.verbose and constant_cols:
            print(f"[SELECTION] Supprimé {len(constant_cols)} features constantes")

        # 2. Supprimer features trop corrélées (priorité LLMFE)
        if self.config.correlation_threshold < 1.0:
            X_clean = self._remove_correlated_features(
                X_clean,
                llmfe_features=llmfe_features,
                threshold=self.config.correlation_threshold,
            )

        # 3. Calculer l'importance des features
        feature_importances = self._calculate_importance(X_clean, y_encoded, is_regression)

        # 4. Sélection par importance si demandé
        if self.config.final_selection != "none":
            # Trier par importance
            sorted_features = sorted(
                feature_importances.keys(),
                key=lambda f: feature_importances[f],
                reverse=True,
            )

            # Appliquer max_features
            if self.config.max_features and len(sorted_features) > self.config.max_features:
                # Garder toutes les features LLMFE en priorité
                llmfe_to_keep = [f for f in sorted_features if f in llmfe_features]
                other_features = [f for f in sorted_features if f not in llmfe_features]

                # Combiner LLMFE + meilleures autres jusqu'à max_features
                remaining_slots = self.config.max_features - len(llmfe_to_keep)
                if remaining_slots > 0:
                    selected_features = llmfe_to_keep + other_features[:remaining_slots]
                else:
                    selected_features = llmfe_to_keep[: self.config.max_features]

                X_clean = X_clean[selected_features]

                if self.config.verbose:
                    print(f"[SELECTION] Limité à {self.config.max_features} features")
                    print(
                        f"            - LLMFE gardées: {len([f for f in selected_features if f in llmfe_features])}"
                    )

        final_features = list(X_clean.columns)

        if self.config.verbose:
            print(f"[SELECTION] Features finales: {len(final_features)}")

        # Reconstruire le DataFrame final
        df_final = X_clean.copy()
        df_final[target_col] = y.values

        return df_final, final_features, feature_importances

    def _remove_correlated_features(
        self,
        X: pd.DataFrame,
        llmfe_features: list[str],
        threshold: float,
    ) -> pd.DataFrame:
        """Supprime les features trop corrélées en gardant LLMFE en priorité."""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()

        for col in upper_tri.columns:
            correlated = upper_tri.index[upper_tri[col] > threshold].tolist()

            for corr_col in correlated:
                # Si les deux sont LLMFE ou les deux ne sont pas LLMFE,
                # garder celui avec le moins de NaN
                col_is_llmfe = col in llmfe_features
                corr_is_llmfe = corr_col in llmfe_features

                if col_is_llmfe and not corr_is_llmfe:
                    # Garder LLMFE, supprimer l'autre
                    to_drop.add(corr_col)
                elif corr_is_llmfe and not col_is_llmfe:
                    # Garder LLMFE, supprimer l'autre
                    to_drop.add(col)
                else:
                    # Même priorité : garder celui avec moins de NaN
                    if X[col].isna().sum() <= X[corr_col].isna().sum():
                        to_drop.add(corr_col)
                    else:
                        to_drop.add(col)

        if to_drop and self.config.verbose:
            print(f"[SELECTION] Supprimé {len(to_drop)} features corrélées (>{threshold})")

        return X.drop(columns=list(to_drop))

    def _calculate_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        is_regression: bool,
    ) -> dict[str, float]:
        """Calcule l'importance des features via RandomForest."""
        if is_regression:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1,
            )

        model.fit(X, y)
        return dict(zip(X.columns, model.feature_importances_, strict=False))

    def _calculate_baseline_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_regression: bool,
    ) -> float:
        """Calcule le score baseline sur les données originales."""
        X_clean = self._prepare_for_eval(X)
        return self._calculate_score(X_clean, y, is_regression)

    def _calculate_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_regression: bool,
    ) -> float:
        """Calcule le score via cross-validation."""
        try:
            X_clean = self._prepare_for_eval(X)

            # Encoder la target si nécessaire
            y_encoded = y
            if not is_regression and y.dtype == "object":
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

            # Modèle
            if is_regression:
                from xgboost import XGBRegressor

                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
                scoring = "neg_root_mean_squared_error"
            else:
                from xgboost import XGBClassifier

                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbosity=0,
                    eval_metric="logloss",
                )
                scoring = "f1_weighted"

            scores = cross_val_score(
                model,
                X_clean,
                y_encoded,
                cv=self.config.cv_folds,
                scoring=scoring,
                n_jobs=-1,
            )

            if "neg_" in scoring:
                return -np.mean(scores)
            return np.mean(scores)

        except Exception:
            return 0.0

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

        # Supprimer infinies
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(0)

        return X_clean

    def _save_results(self, output_dir: Path) -> None:
        """Sauvegarde les résultats."""
        # Sauvegarder le DataFrame transformé
        if self._df_transformed is not None:
            train_path = output_dir / "train_hybrid.parquet"
            self._df_transformed.to_parquet(train_path)

        # Sauvegarder le rapport JSON
        if self._result is not None:
            report_path = output_dir / "hybrid_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self._result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        if self.config.verbose:
            print(f"\n[HYBRID] Résultats sauvegardés dans: {output_dir}")

    def _print_header(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_regression: bool,
    ) -> None:
        """Affiche l'en-tête."""
        print("\n" + "=" * 70)
        print("       FEATURE ENGINEERING HYBRIDE (LLMFE + DFS)")
        print("=" * 70)
        print(f"  Projet        : {self.project_name}")
        print(f"  Dataset       : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"  Target        : {target_col}")
        print(f"  Type          : {'Régression' if is_regression else 'Classification'}")
        print(f"  LLMFE activé  : {'Oui' if self.config.enable_llmfe else 'Non'}")
        print(f"  DFS activé    : {'Oui' if self.config.enable_dfs else 'Non'}")
        print(f"  Max features  : {self.config.max_features or 'Illimité'}")
        print("=" * 70)

    def _print_summary(self) -> None:
        """Affiche le résumé final."""
        if self._result is None:
            return

        r = self._result
        print("\n" + "=" * 70)
        print("       RÉSUMÉ FEATURE ENGINEERING HYBRIDE")
        print("=" * 70)
        print(f"  Features originales  : {r.n_original}")
        print(f"  + LLMFE              : +{r.n_llmfe_added}")
        print(f"  + DFS                : +{r.n_dfs_added}")
        print(f"  = Total avant select : {r.n_original + r.n_llmfe_added + r.n_dfs_added}")
        print(f"  - Supprimées         : -{r.n_dropped}")
        print(f"  = Features finales   : {r.n_final}")
        print("─" * 70)
        print(f"  Score baseline       : {r.baseline_score:.4f}")
        print(
            f"  Score après LLMFE    : {r.llmfe_score:.4f} ({r.llmfe_score - r.baseline_score:+.4f})"
        )
        print(f"  Score après DFS      : {r.dfs_score:.4f} ({r.dfs_score - r.llmfe_score:+.4f})")
        print(f"  Score final          : {r.final_score:.4f}")
        print(f"  Amélioration totale  : {r.improvement_pct:+.2f}%")
        print("─" * 70)
        print(f"  Temps LLMFE          : {r.llmfe_time_seconds:.1f}s")
        print(f"  Temps DFS            : {r.dfs_time_seconds:.1f}s")
        print(f"  Temps sélection      : {r.selection_time_seconds:.1f}s")
        print(f"  Temps total          : {r.total_time_seconds:.1f}s")
        print("=" * 70)

        # Top 10 features par importance
        if r.feature_importances:
            print("\n  Top 10 features par importance:")
            sorted_features = sorted(
                r.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            for i, (feat, imp) in enumerate(sorted_features, 1):
                source = (
                    "LLMFE"
                    if feat in r.llmfe_features
                    else "DFS"
                    if feat in r.dfs_features
                    else "ORIG"
                )
                print(f"    {i:2}. [{source:5}] {feat}: {imp:.4f}")

    def get_transformed_data(self) -> pd.DataFrame | None:
        """Retourne le DataFrame transformé."""
        return self._df_transformed

    def get_result(self) -> HybridResult | None:
        """Retourne le résultat complet."""
        return self._result


def run_hybrid_fe(
    df_train: pd.DataFrame,
    target_col: str,
    project_name: str,
    config: HybridConfig | str = "default",
    df_test: pd.DataFrame | None = None,
    is_regression: bool = False,
    analyse_path: Path | str | None = None,
    task_description: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, HybridResult]:
    """
    Fonction raccourcie pour exécuter le Feature Engineering Hybride.

    Args:
        df_train: DataFrame d'entraînement
        target_col: Colonne cible
        project_name: Nom du projet
        config: Configuration ou nom d'une config prédéfinie
        df_test: DataFrame de test (optionnel)
        is_regression: Type de tâche
        analyse_path: Chemin vers le rapport d'analyse
        task_description: Description de la tâche
        verbose: Afficher les logs

    Returns:
        Tuple (DataFrame transformé, HybridResult)

    Example:
        >>> df_transformed, result = run_hybrid_fe(
        ...     df_train=train_df,
        ...     target_col="Survived",
        ...     project_name="titanic_hybrid",
        ... )
    """
    # Charger la config si c'est un nom
    if isinstance(config, str):
        from src.feature_engineering.hybrid.config import get_hybrid_config

        config = get_hybrid_config(config)

    config.verbose = verbose

    engineer = HybridFeatureEngineer(project_name=project_name, config=config)
    result = engineer.run(
        df_train=df_train,
        target_col=target_col,
        df_test=df_test,
        is_regression=is_regression,
        analyse_path=analyse_path,
        task_description=task_description,
    )

    return engineer.get_transformed_data(), result
