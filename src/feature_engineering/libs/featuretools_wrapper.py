from __future__ import annotations

# featuretools_wrapper.py
import featuretools as ft
import pandas as pd


class AutoFeatureEngineer:
    """
    Classe pour effectuer un feature engineering automatique avec Featuretools.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_entity: str = "main",
        index: str = None,
        time_index: str = None,
        max_depth: int = 2,
        agg_primitives: list = None,
        trans_primitives: list = None,
        ignore_columns: list = None,
        verbose: bool = True,
    ):
        """
        Initialisation.

        Paramètres :
        - df : DataFrame source
        - target_entity : nom de l'entité principale
        - index : nom de la colonne identifiant unique
        - time_index : nom de la colonne de type datetime
        - max_depth : profondeur maximale des features
        - agg_primitives : liste de primitives d'agrégation
        - trans_primitives : liste de primitives de transformation
        - ignore_columns : colonnes à ignorer
        - verbose : bool, affichage du processus
        """
        self.df = df.copy()
        self.target_entity = target_entity
        self.index = index
        self.time_index = time_index
        self.max_depth = max_depth
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.ignore_columns = ignore_columns or []
        self.verbose = verbose

        self.es = ft.EntitySet(id="AutoES")
        self.features_df = None

    def create_entity(self):
        """
        Crée l'entité principale dans l'EntitySet.
        """
        if self.verbose:
            print(
                f"[INFO] Création de l'entité '{self.target_entity}' avec index '{self.index}' et time_index '{self.time_index}'"
            )
        self.es = self.es.add_dataframe(
            dataframe_name=self.target_entity,
            dataframe=self.df,
            index=self.index,
            time_index=self.time_index,
            logical_types=None,
            make_index=(self.index is None),
        )

    def run_dfs(self):
        """
        Lance le Deep Feature Synthesis (DFS) pour créer automatiquement les features.
        """
        if self.verbose:
            print("[INFO] Exécution de DFS...")
        self.features_df, self.feature_defs = ft.dfs(
            entityset=self.es,
            target_dataframe_name=self.target_entity,
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            max_depth=self.max_depth,
            verbose=self.verbose,
            features_only=False,
            ignore_columns={self.target_entity: self.ignore_columns}
            if self.ignore_columns
            else None,
        )
        if self.verbose:
            print(f"[INFO] DFS terminé. {self.features_df.shape[1]} features générées.")

    def get_features(self):
        """
        Retourne le DataFrame avec les features créées.
        """
        if self.features_df is None:
            raise ValueError("DFS n'a pas encore été exécuté. Appelez run_dfs() d'abord.")
        return self.features_df

    def save_features(self, path: str):
        """
        Sauvegarde le DataFrame des features générées.
        """
        if self.features_df is None:
            raise ValueError("DFS n'a pas encore été exécuté. Appelez run_dfs() d'abord.")
        self.features_df.to_csv(path, index=False)
        if self.verbose:
            print(f"[INFO] Features sauvegardées dans {path}")
