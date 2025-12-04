# src/features_engineering/lib_existante/feature_engine.py
# VERSION FINALE – ZÉRO ERREUR – feature-engine 1.9.3

import warnings

import numpy as np
import pandas as pd
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.encoding import (
    CountFrequencyEncoder,
    MeanEncoder,
    OneHotEncoder,
    RareLabelEncoder,
    WoEEncoder,
)
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# =============================================
# TRANSFORMERS MANUELS (plus stables que MathFeatures 1.9.3)
# =============================================
class PairwiseRatios(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, max_pairs=50):
        self.variables = variables
        self.max_pairs = max_pairs
        self.pairs_ = []

    def fit(self, X, y=None):
        if self.variables is None:
            self.variables = X.select_dtypes(include=np.number).columns.tolist()
        cols = [c for c in self.variables if c in X.columns]
        if len(cols) < 2:
            return self

        # Sélection des meilleures paires (corrélation avec y)
        if y is not None:
            corrs = np.abs([np.corrcoef(X[c].fillna(0), y.fillna(0))[0, 1] for c in cols])
            top_idx = np.argsort(corrs)[-min(15, len(cols)) :]
            selected_cols = [cols[i] for i in top_idx]
        else:
            selected_cols = cols[:15]

        import itertools

        self.selected_pairs = list(itertools.combinations(selected_cols, 2))[: self.max_pairs]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.variables:
            if c not in X.columns:
                continue
            col = X[c].fillna(0)
            X[f"{c}_log"] = np.log1p(col)
            X[f"{c}_sqrt"] = np.sqrt(np.abs(col))
            X[f"{c}_square"] = col**2
            X[f"{c}_cube"] = col**3
            X[f"{c}_inv"] = 1 / (col + 1e-6)
            X[f"{c}_round"] = col.round()
            X[f"{c}_sin"] = np.sin(col)
            X[f"{c}_cos"] = np.cos(col)

        # Interactions puissantes
        for c1, c2 in self.selected_pairs:
            if c1 in X.columns and c2 in X.columns:
                v1, v2 = X[c1].fillna(0), X[c2].fillna(0)
                X[f"{c1}_{c2}_prod"] = v1 * v2
                X[f"{c1}_{c2}_ratio"] = v1 / (v2 + 1e-6)
                X[f"{c1}_{c2}_diff"] = v1 - v2
                X[f"{c1}_{c2}_sum"] = v1 + v2

        # Stats globales
        num_cols = [c for c in X.columns if X[c].dtype in ["float64", "int64"]]
        if len(num_cols) >= 2:
            X["num_mean"] = X[num_cols].mean(axis=1)
            X["num_std"] = X[num_cols].std(axis=1)
            X["num_skew"] = X[num_cols].skew(axis=1)
            X["num_kurt"] = X[num_cols].kurt(axis=1)
            X["num_range"] = X[num_cols].max(axis=1) - X[num_cols].min(axis=1)
        return X


class ThresholdFeatures(BaseEstimator, TransformerMixin):
    """
    Crée des features booléennes basées sur des seuils :
      - Valeurs absolues (ex: Age > 10)
      - Valeurs relatives (ex: > median, > quantile 0.75)
    """

    def __init__(self, variables=None, thresholds=None, quantiles=[0.25, 0.5, 0.75]):
        """
        variables : liste des colonnes numériques
        thresholds : dict {col: [val1, val2]} pour définir des seuils custom
        quantiles  : liste des quantiles à utiliser si thresholds=None
        """
        self.variables = variables
        self.thresholds = thresholds
        self.quantiles = quantiles
        self.thresholds_computed_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        if self.variables is None:
            self.variables = X.select_dtypes(include=np.number).columns.tolist()

        # calculer les seuils
        self.thresholds_computed_ = {}
        for col in self.variables:
            if self.thresholds and col in self.thresholds:
                self.thresholds_computed_[col] = self.thresholds[col]
            else:
                self.thresholds_computed_[col] = X[col].quantile(self.quantiles).tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col, ths in self.thresholds_computed_.items():
            if col not in X.columns:
                continue
            for th in ths:
                X[f"{col}_gt_{th}"] = (X[col] > th).astype(int)
                X[f"{col}_lt_{th}"] = (X[col] < th).astype(int)
        return X


class AggFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        if self.variables is None:
            self.variables = X.select_dtypes(include=np.number).columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.variables if c in X.columns]
        if len(cols) >= 2:
            X["num_mean"] = X[cols].mean(axis=1)
            X["num_std"] = X[cols].std(axis=1)
            X["num_max"] = X[cols].max(axis=1)
            X["num_min"] = X[cols].min(axis=1)
            X["num_sum"] = X[cols].sum(axis=1)
        return X


# =============================================
# CLASSE PRINCIPALE
# =============================================
class AutoFeatureGenerator:
    def __init__(
        self,
        target: str,
        rare_tol: float = 0.01,
        encoding_strategy: str = "mean",
        winsorize: float = 0.05,
        add_cyclical: bool = True,
        add_pairwise: bool = True,
        max_pairs: int = 40,
    ):
        self.target = target
        self.rare_tol = rare_tol
        self.encoding_strategy = encoding_strategy
        self.winsorize = winsorize
        self.add_cyclical = add_cyclical
        self.add_pairwise = add_pairwise
        self.max_pairs = max_pairs

        self.pipeline = None
        self.n_features_in_ = None
        self.n_features_out_ = None

    def _detect_columns(self, X):
        num = X.select_dtypes(include=np.number).columns.tolist()
        cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        dt = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        return num, cat, dt

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = X.copy()
        y = y.copy() if y is not None else None

        num_cols, cat_cols, dt_cols = self._detect_columns(X)
        steps = []

        # 1. Imputation
        if num_cols:
            steps.append(
                ("imp_num", MeanMedianImputer(imputation_method="median", variables=num_cols))
            )
        if cat_cols:
            steps.append(("imp_cat", CategoricalImputer(fill_value="Missing", variables=cat_cols)))

        # 2. Rare labels
        if cat_cols:
            steps.append(("rare", RareLabelEncoder(tol=self.rare_tol, variables=cat_cols)))

        # 3. Encodage
        if cat_cols:
            if self.encoding_strategy == "mean":
                steps.append(("encoder", MeanEncoder(variables=cat_cols)))
            elif self.encoding_strategy == "woe":
                steps.append(("encoder", WoEEncoder(variables=cat_cols)))
            elif self.encoding_strategy == "frequency":
                steps.append(("encoder", CountFrequencyEncoder(variables=cat_cols)))
            else:
                steps.append(("encoder", OneHotEncoder(drop_last=True, variables=cat_cols)))

        # 4. Datetime + Cyclical
        if dt_cols:
            if self.add_cyclical:
                steps.append(("cyclical", CyclicalFeatures(variables=dt_cols, drop_original=True)))
            steps.append(
                (
                    "dt",
                    DatetimeFeatures(
                        variables=dt_cols,
                        features_to_extract=[
                            "month",
                            "day_of_week",
                            "hour",
                            "weekend",
                            "day_of_year",
                        ],
                    ),
                )
            )

        # 5. Features numériques puissantes (remplace MathFeatures buggé)
        if num_cols and len(num_cols) >= 2:
            steps.append(("agg", AggFeatures(variables=num_cols)))
            if self.add_pairwise:
                steps.append(
                    ("pairwise", PairwiseRatios(variables=num_cols, max_pairs=self.max_pairs))
                )

        if num_cols:
            # Threshold / boolean features
            steps.append(
                ("threshold", ThresholdFeatures(variables=num_cols, quantiles=[0.25, 0.5, 0.75]))
            )

        # 6. Winsorization
        if num_cols and self.winsorize > 0:
            steps.append(
                (
                    "winsor",
                    Winsorizer(
                        capping_method="gaussian",
                        tail="both",
                        fold=self.winsorize,
                        variables=num_cols,
                    ),
                )
            )

        # 7. Nettoyage
        steps += [
            ("drop_const", DropConstantFeatures(tol=0.98)),
            ("drop_dupl", DropDuplicateFeatures()),
            ("drop_corr", DropCorrelatedFeatures(threshold=0.95)),
        ]

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)

        self.n_features_in_ = X.shape[1]
        X_out = self.pipeline.transform(X)
        self.n_features_out_ = X_out.shape[1]

        print("AUTO FEATURE GENERATOR 100 % OK")
        print(
            f"   {self.n_features_in_} → {self.n_features_out_} features (+{self.n_features_out_ - self.n_features_in_})"
        )

        return self

    def transform(self, X):
        if self.pipeline is None:
            raise ValueError("Fit d'abord !")
        return self.pipeline.transform(X.copy())

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
