# ============================================================
# FeatureCorrelationAnalyzer — Version Ultra Complète
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# Imports optionnels - ces librairies peuvent ne pas être installées
try:
    from minepy import MINE
    HAS_MINEPY = True
except ImportError:
    HAS_MINEPY = False
    print("[CORRELATIONS] minepy non installé - MIC sera ignoré")

try:
    import phik
    HAS_PHIK = True
except ImportError:
    HAS_PHIK = False
    print("[CORRELATIONS] phik non installé - PhiK sera ignoré")

from joblib import Parallel, delayed
from typing import Optional, List


# ============================================================
#                MAIN ANALYZER CLASS
# ============================================================

class FeatureCorrelationAnalyzer:
    """
    Analyse complète des corrélations pour dataset ML :
    - Pearson / Spearman / Kendall
    - Mutual Information (classification/regression)
    - MIC (Maximal Information Coefficient)
    - PhiK
    - Heatmap combinée
    - Scoring automatique
    """

    def __init__(self, df: pd.DataFrame, target_col: str, task: str = "classification"):
        if target_col not in df.columns:
            raise ValueError(f"Colonne cible '{target_col}' non trouvée dans dataframe.")

        self.df = df.copy()
        self.target_col = target_col
        self.task = task
        self.X = None
        self.y = self.df[target_col]

    # --------------------------------------------------------
    # Feature prep
    # --------------------------------------------------------
    def _prepare_features(self):
        """
        Convertit toutes les variables en numériques :
        - Catégorielles → codes entiers
        - NaN numériques → médianes
        - NaN catégorielles → "missing"
        """
        X = self.df.drop(columns=[self.target_col]).copy()

        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        X[cat_cols] = X[cat_cols].astype("string").fillna("missing")

        for c in cat_cols:
            X[c] = LabelEncoder().fit_transform(X[c])

        self.X = X
        return X

    # ============================================================
    #                 CLASSIC CORRELATIONS
    # ============================================================

    def compute_classical_correlations(self):
        """Retourne Pearson / Spearman / Kendall."""
        X = self._prepare_features()

        pearson = X.corrwith(self.y, method="pearson")
        spearman = X.corrwith(self.y, method="spearman")
        kendall = X.corrwith(self.y, method="kendall")

        df_corr = pd.DataFrame({
            "feature": X.columns,
            "pearson": pearson.values,
            "spearman": spearman.values,
            "kendall": kendall.values,
        })

        return df_corr

    # ============================================================
    #                 MUTUAL INFORMATION
    # ============================================================

    def compute_mutual_info(self):
        X = self._prepare_features()
        discrete = (X.dtypes != float).values

        if self.task == "classification":
            mi = mutual_info_classif(X, self.y, discrete_features=discrete, random_state=42)
        else:
            mi = mutual_info_regression(X, self.y, discrete_features=discrete, random_state=42)

        return pd.DataFrame({
            "feature": X.columns,
            "mutual_info": mi
        })

    # ============================================================
    #                       MIC
    # ============================================================

    def compute_mic_matrix(self):
        """Calcule MIC entre chaque feature et la cible."""
        if not HAS_MINEPY:
            # Retourner un DataFrame vide si minepy n'est pas installé
            X = self._prepare_features()
            return pd.DataFrame({
                "feature": X.columns,
                "mic": [0.0] * len(X.columns)
            })

        X = self._prepare_features()
        mic_scores = []

        mine = MINE()

        for col in X.columns:
            mine.compute_score(X[col], self.y)
            mic_scores.append(mine.mic())

        return pd.DataFrame({
            "feature": X.columns,
            "mic": mic_scores
        })

    # ============================================================
    #                        PhiK
    # ============================================================

    def compute_phik(self):
        """Utilise phik pour corrélations robustes catégoriel/numérique."""
        if not HAS_PHIK:
            # Retourner un DataFrame vide si phik n'est pas installé
            X = self._prepare_features()
            return pd.DataFrame({
                "feature": X.columns,
                "phik": [0.0] * len(X.columns)
            })

        X = self._prepare_features()
        df_full = pd.concat([X, self.y], axis=1)

        phik_matrix = df_full.phik_matrix(interval_cols=X.columns.tolist())
        target_row = phik_matrix.loc[self.target_col, X.columns]

        return pd.DataFrame({
            "feature": X.columns,
            "phik": target_row.values
        })

    # ============================================================
    #              PARALLEL MIC (10× plus rapide)
    # ============================================================

    def parallel_compute_mic(self, n_jobs=-1):
        if not HAS_MINEPY:
            X = self._prepare_features()
            return pd.DataFrame({
                "feature": X.columns,
                "mic": [0.0] * len(X.columns)
            })

        X = self._prepare_features()

        def mic_one(col):
            mine = MINE()
            mine.compute_score(X[col], self.y)
            return col, mine.mic()

        results = Parallel(n_jobs=n_jobs)(
            delayed(mic_one)(col) for col in X.columns
        )

        return pd.DataFrame(results, columns=["feature", "mic"])

    # ============================================================
    #             COMBINED FEATURE SCORING
    # ============================================================

    def combined_feature_score(self, normalize=True):
        """Combine plusieurs scores en un score unique."""

        corr = self.compute_classical_correlations()
        mi = self.compute_mutual_info()

        df = corr.merge(mi, on="feature")

        # Ajouter MIC si disponible
        if HAS_MINEPY:
            mic = self.compute_mic_matrix()
            df = df.merge(mic, on="feature")

        # Ajouter PhiK si disponible
        if HAS_PHIK:
            phik_df = self.compute_phik()
            df = df.merge(phik_df, on="feature")

        # Colonnes disponibles pour le score
        score_cols = ["pearson", "spearman", "kendall", "mutual_info"]
        if HAS_MINEPY:
            score_cols.append("mic")
        if HAS_PHIK:
            score_cols.append("phik")

        # Option : normalisation 0-1
        if normalize:
            for col in score_cols:
                if col in df.columns:
                    df[col] = df[col].abs()
                    df[col] = df[col] / (df[col].max() + 1e-9)

        # Score final (moyenne des colonnes disponibles)
        df["combined_score"] = df[score_cols].mean(axis=1)

        return df.sort_values(by="combined_score", ascending=False)

    # ============================================================
    #                     HEATMAP MULTI-MATRICE
    # ============================================================

    def plot_multi_heatmap(self, save_path=None):

        corr_df = self.compute_classical_correlations().set_index("feature")
        mi_df = self.compute_mutual_info().set_index("feature")
        mic_df = self.compute_mic_matrix().set_index("feature")
        phik_df = self.compute_phik().set_index("feature")

        full = corr_df.join(mi_df).join(mic_df).join(phik_df)

        plt.figure(figsize=(12, 8))
        sns.heatmap(full, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Heatmap Multi-Matrice (Corrélations & Information)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()


def use_all(df,target_col,task):

        # 1. Instanciation
    analyzer = FeatureCorrelationAnalyzer(df, target_col=target_col,task=task)

    # 2. Corrélations
    corr = analyzer.compute_classical_correlations()
    print("\n=== Corrélations classiques ===")
    print(corr)

    # 3. Mutual information
    mi = analyzer.compute_mutual_info()
    print("\n=== Mutual Information ===")
    print(mi)

    # 4. MIC (minepy)
    mic = analyzer.compute_mic_matrix()
    print("\n=== MIC ===")
    print(mic)

    # 5. PhiK
    pk = analyzer.compute_phik()
    print("\n=== PhiK ===")
    print(pk)

    # 6. MIC parallèle
    mic_fast = analyzer.parallel_compute_mic()
    print("\n=== MIC parallèle ===")
    print(mic_fast)

    # 7. Score combiné
    scores = analyzer.combined_feature_score()
    print("\n=== Score combiné ===")
    print(scores)


    # 8. Heatmap globale
    # analyzer.plot_multi_heatmap()

    return scores



import numpy as np
import pandas as pd

def get_top_features(corr_df, n=20, penalize_ids=True, id_threshold=0.9, id_factor=0.1):
    """
    Retourne les n meilleures features selon un score combiné
    utilisant Pearson, Spearman, Kendall, Mutual Info, MIC et Phik.
    
    Penalise fortement les colonnes ressemblant à des IDs (beaucoup de valeurs uniques).

    corr_df : DataFrame produit par `compute_all_correlations`
    n       : nombre de features à garder
    penalize_ids : si True, baisse les scores des colonnes ressemblant à des IDs
    id_threshold : fraction unique/total pour considérer comme ID
    id_factor    : facteur multiplicatif pour pénaliser les IDs

    Retourne : DataFrame trié + liste des noms de features retenues
    """
    df = corr_df.copy()
    df = df.replace({None: np.nan})

    metric_cols = [c for c in ["pearson", "spearman", "kendall",
                               "mutual_info", "mic", "phik"]
                   if c in df.columns]

    df[metric_cols] = df[metric_cols].fillna(0)

    # Normalisation min-max
    for col in metric_cols:
        col_min, col_max = df[col].min(), df[col].max()
        df[col] = (df[col] - col_min) / (col_max - col_min) if col_max != col_min else 0

    # Score global pondéré
    df["global_score"] = (
        0.20 * df.get("pearson", 0).abs() +
        0.20 * df.get("spearman", 0).abs() +
        0.15 * df.get("kendall", 0).abs() +
        0.25 * df.get("mutual_info", 0) +
        0.10 * df.get("mic", 0) +
        0.10 * df.get("phik", 0)
    )

    # Détecter et pénaliser les IDs
    if penalize_ids:
        def id_penalty(row):
            feature_name = str(row["feature"]).lower()
            # Si le nom ressemble à un ID
            if "id" in feature_name or "code" in feature_name:
                return id_factor
            # Si la fraction de valeurs uniques est très haute
            if "unique_frac" in row:
                return id_factor if row["unique_frac"] >= id_threshold else 1
            return 1

        df["penalty"] = df.apply(id_penalty, axis=1)
        df["global_score"] *= df["penalty"]
        df.drop(columns="penalty", inplace=True)

    df_sorted = df.sort_values(by="global_score", ascending=False)
    top_features = df_sorted["feature"].head(n).tolist()

    return df_sorted.head(n), top_features

