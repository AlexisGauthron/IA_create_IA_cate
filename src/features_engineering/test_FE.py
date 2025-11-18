from __future__ import annotations

from typing import Optional, Sequence, Literal, Dict, Any
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# --- feature-engine ---
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper

# --- dirty_cat ---
from dirty_cat import TableVectorizer

# --- autofeat (optionnel) ---
from autofeat import AutoFeatRegressor, AutoFeatClassifier


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering 'générique' pour ton pipeline IA_create_IA.

    Capable de :
      - FE tabulaire (numérique + catégoriel) via feature-engine
      - Encodage texte / colonnes "sales" via dirty_cat.TableVectorizer
      - Génération de features non-linéaires optionnelle via autofeat

    Paramètres principaux
    ---------------------
    numeric_cols : liste ou None
        Colonnes numériques. Si None, déduites automatiquement.
    categorical_cols : liste ou None
        Colonnes catégorielles. Si None, déduites automatiquement
        (tous les object/category qui ne sont PAS dans text_cols).
    text_cols : liste ou None
        Colonnes texte à envoyer dans TableVectorizer (dirty_cat).
        Si None, aucun texte n'est traité.

    use_tabular_fe : bool
        Active le bloc feature-engine (imputation, encodage, scaling, outliers).
    use_dirty_cat : bool
        Active le bloc dirty_cat.TableVectorizer sur text_cols.
    use_autofeat : bool
        Si True, passe les features combinées dans AutoFeat pour générer
        et sélectionner des features non-linéaires.

    autofeat_problem_type : {"classification", "regression"}
        Type de problème passé à AutoFeat (choix du modèle interne).
    autofeat_kwargs : dict ou None
        Paramètres supplémentaires pour AutoFeatRegressor/Classifier
        (ex: {"feat_eng_steps": 2, "max_steps": 3, "n_jobs": -1}).

    handle_outliers : bool
        Active la winsorisation des colonnes numériques.
    scale_numeric : bool
        Active la standardisation des colonnes numériques.
    rare_label_tol : float
        Seuil de RareLabelEncoder pour les catégories rares.
    """

    def __init__(
        self,
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        text_cols: Optional[Sequence[str]] = None,
        *,
        use_tabular_fe: bool = True,
        use_dirty_cat: bool = True,
        use_autofeat: bool = False,
        autofeat_problem_type: Literal["classification", "regression"] = "classification",
        autofeat_kwargs: Optional[Dict[str, Any]] = None,
        handle_outliers: bool = True,
        scale_numeric: bool = True,
        rare_label_tol: float = 0.05,
    ) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols

        self.use_tabular_fe = use_tabular_fe
        self.use_dirty_cat = use_dirty_cat
        self.use_autofeat = use_autofeat
        self.autofeat_problem_type = autofeat_problem_type
        self.autofeat_kwargs = autofeat_kwargs or {}

        self.handle_outliers = handle_outliers
        self.scale_numeric = scale_numeric
        self.rare_label_tol = rare_label_tol

        # Attributs qui seront définis au fit
        self.tabular_pipe_: Optional[Pipeline] = None
        self.table_vec_: Optional[TableVectorizer] = None
        self.autofeat_: Optional[AutoFeatRegressor | AutoFeatClassifier] = None

        self.numeric_cols_: Sequence[str] = []
        self.categorical_cols_: Sequence[str] = []
        self.text_cols_: Sequence[str] = []

        self.base_feature_names_: Sequence[str] = []
        self.feature_names_: Sequence[str] = []
        self.n_features_out_: int = 0

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------
    def _infer_column_types(self, X: pd.DataFrame) -> None:
        """Infère numeric / catégoriel / texte si non fournis."""
        # Numériques
        if self.numeric_cols is None:
            self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols_ = list(self.numeric_cols)

        # Texte
        if self.text_cols is None:
            self.text_cols_ = []
        else:
            self.text_cols_ = list(self.text_cols)

        # Catégorielles = object/category qui ne sont pas dans text_cols
        if self.categorical_cols is None:
            possible_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
            self.categorical_cols_ = [c for c in possible_cat if c not in self.text_cols_]
        else:
            self.categorical_cols_ = list(self.categorical_cols)

    def _fit_tabular_pipe(self, X: pd.DataFrame) -> None:
        """Construit et fit le pipeline feature-engine sur X."""
        if not self.use_tabular_fe:
            self.tabular_pipe_ = None
            return

        steps = []

        # Numériques
        if len(self.numeric_cols_) > 0:
            steps.append(
                ("impute_num", MeanMedianImputer(
                    variables=list(self.numeric_cols_),
                    imputation_method="median",
                ))
            )

            if self.handle_outliers:
                steps.append(
                    ("winsor_num", Winsorizer(
                        capping_method="gaussian",
                        tail="both",
                        fold=3.0,
                        variables=list(self.numeric_cols_),
                    ))
                )

            if self.scale_numeric:
                steps.append(
                    ("scale_num", SklearnTransformerWrapper(
                        transformer=StandardScaler(),
                        variables=list(self.numeric_cols_),
                    ))
                )

        # Catégorielles
        if len(self.categorical_cols_) > 0:
            steps.append(
                ("impute_cat", CategoricalImputer(
                    variables=list(self.categorical_cols_),
                    fill_value="__MISSING__",
                ))
            )

            steps.append(
                ("rare_cat", RareLabelEncoder(
                    variables=list(self.categorical_cols_),
                    tol=self.rare_label_tol,
                    n_categories=1,
                ))
            )

            steps.append(
                ("onehot_cat", OneHotEncoder(
                    variables=list(self.categorical_cols_),
                    drop_last=True,
                ))
            )

        if len(steps) == 0:
            # Pas de FE tabulaire à faire
            self.tabular_pipe_ = None
            return

        self.tabular_pipe_ = Pipeline(steps=steps)
        self.tabular_pipe_.fit(X)

    def _fit_table_vectorizer(self, X: pd.DataFrame) -> None:
        """Fit TableVectorizer sur les colonnes texte (dirty_cat)."""
        if not self.use_dirty_cat or len(self.text_cols_) == 0:
            self.table_vec_ = None
            return

        self.table_vec_ = TableVectorizer()
        self.table_vec_.fit(X[self.text_cols_])

    def _build_base_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applique :
          - pipeline feature-engine (tabulaire) sur X
          - TableVectorizer sur text_cols
        et concatène tout dans un numpy array.
        Met à jour base_feature_names_ à la première utilisation.
        """
        X = X.copy()

        # 1) FE tabulaire
        if self.tabular_pipe_ is not None:
            X_tab = self.tabular_pipe_.transform(X)
        else:
            X_tab = X

        # On enlève les colonnes texte de la partie tabulaire
        if len(self.text_cols_) > 0:
            X_tab_no_text = X_tab.drop(columns=self.text_cols_, errors="ignore")
        else:
            X_tab_no_text = X_tab

        parts = []
        names: list[str] = []

        if X_tab_no_text.shape[1] > 0:
            parts.append(X_tab_no_text.to_numpy())
            names.extend([str(c) for c in X_tab_no_text.columns])

        # 2) Texte / colonnes sales via dirty_cat
        if self.table_vec_ is not None and len(self.text_cols_) > 0:
            Xt = self.table_vec_.transform(X[self.text_cols_])
            # Converter sparse -> dense si besoin
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            parts.append(Xt)
            tv_names = list(self.table_vec_.get_feature_names_out())
            names.extend(tv_names)

        if len(parts) == 0:
            X_comb = np.empty((len(X), 0))
        else:
            X_comb = np.hstack(parts)

        # Si on n'a pas encore fixé les noms de base, on le fait ici
        if not hasattr(self, "base_feature_names_") or len(self.base_feature_names_) == 0:
            self.base_feature_names_ = names

        return X_comb

    # ------------------------------------------------------------------
    # API sklearn
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Fit le feature engineering.
        - Infère les colonnes si besoin
        - Fit feature-engine sur tabulaire
        - Fit dirty_cat sur texte
        - (Optionnel) Fit AutoFeat sur les features combinées
        """
        X = pd.DataFrame(X).copy()

        # 1) Inférer les types de colonnes
        self._infer_column_types(X)

        # 2) Fit FE tabulaire
        self._fit_tabular_pipe(X)

        # 3) Fit FE texte
        self._fit_table_vectorizer(X)

        # 4) Construire les features de base
        X_base = self._build_base_features(X)

        # 5) AutoFeat optionnel
        if self.use_autofeat:
            if y is None:
                raise ValueError("y est requis lorsque use_autofeat=True")

            X_base_df = pd.DataFrame(X_base, columns=self.base_feature_names_)

            if self.autofeat_problem_type == "classification":
                self.autofeat_ = AutoFeatClassifier(**self.autofeat_kwargs)
            else:
                self.autofeat_ = AutoFeatRegressor(**self.autofeat_kwargs)

            self.autofeat_.fit(X_base_df, y)

            # On passe une fois pour récupérer les noms finaux
            X_af = self.autofeat_.transform(X_base_df)
            if hasattr(X_af, "columns"):
                self.feature_names_ = [str(c) for c in X_af.columns]
                self.n_features_out_ = X_af.shape[1]
            else:
                # fallback si AutoFeat renvoie un array
                self.feature_names_ = [f"autofeat_{i}" for i in range(X_af.shape[1])]
                self.n_features_out_ = X_af.shape[1]
        else:
            self.autofeat_ = None
            self.feature_names_ = list(self.base_feature_names_)
            self.n_features_out_ = len(self.feature_names_)

        return self

    def transform(self, X):
        """
        Applique le feature engineering sur de nouvelles données.
        Renvoie un numpy array de shape (n_samples, n_features_out_).
        """
        check_is_fitted(self, ["feature_names_", "n_features_out_"])

        X = pd.DataFrame(X).copy()

        # 1) Features de base (tabulaire + texte)
        X_base = self._build_base_features(X)

        # 2) AutoFeat optionnel
        if self.autofeat_ is not None:
            X_base_df = pd.DataFrame(X_base, columns=self.base_feature_names_)
            X_af = self.autofeat_.transform(X_base_df)
            if hasattr(X_af, "values"):
                return X_af.values
            return np.asarray(X_af)
        else:
            return X_base

    def get_feature_names_out(self) -> np.ndarray:
        """Renvoie les noms des features après FE (utile pour debug / interprétation)."""
        check_is_fitted(self, ["feature_names_"])
        return np.array(self.feature_names_, dtype=object)
