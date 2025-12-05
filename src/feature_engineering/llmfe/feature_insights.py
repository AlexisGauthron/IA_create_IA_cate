# src/feature_engineering/llmfe/feature_insights.py
"""
Module pour charger et gérer les insights sur les features.
Utilise TOUJOURS le module src/analyse/ comme source unique de vérité.

Deux modes de fonctionnement:
1. Charger depuis un JSON d'analyse existant (from_json)
2. Lancer l'analyse automatiquement et charger le résultat (from_analyse)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


@dataclass
class FeatureInsight:
    """Insight sur une feature individuelle."""

    name: str
    role: str = "feature"  # feature, target, id, text, timestamp
    inferred_type: str = "unknown"  # numeric, categorical_low, categorical_high, text, etc.

    # Stats de base
    n_rows: int = 0
    n_missing: int = 0
    missing_rate: float = 0.0
    n_unique: int = 0
    unique_ratio: float = 0.0

    # Stats numériques (si applicable)
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None

    # Stats catégorielles (si applicable)
    categories: list[str] = field(default_factory=list)
    top_categories: list[dict[str, Any]] = field(default_factory=list)

    # Corrélation avec la cible (calculée par src/analyse/correlation/)
    correlation: Optional[float] = None  # Pearson
    correlation_spearman: Optional[float] = None
    mutual_info: Optional[float] = None
    combined_score: Optional[float] = None  # Score global

    # Flags et hints (générés par src/analyse/)
    flags: list[str] = field(default_factory=list)  # ID_LIKE, HIGH_CARDINALITY, CONSTANT
    fe_hints: list[str] = field(default_factory=list)  # candidate_for_scaling, use_target_encoding
    notes: list[str] = field(default_factory=list)

    # Description métier (si disponible via LLM)
    description: Optional[str] = None


class FeatureInsights:
    """
    Gestionnaire des insights sur les features.

    IMPORTANT: Cette classe ne calcule JAMAIS les données elle-même.
    Elle charge toujours depuis src/analyse/ (soit un JSON existant, soit en lançant l'analyse).
    """

    def __init__(
        self,
        features: dict[str, FeatureInsight],
        target_name: Optional[str] = None,
        analyse_path: Optional[Path] = None,
    ):
        self.features = features
        self.target_name = target_name
        self.analyse_path = analyse_path  # Chemin du JSON source

    @classmethod
    def from_json(cls, json_path: str | Path) -> FeatureInsights:
        """
        Charge les insights depuis un fichier JSON d'analyse (report_stats.json).

        Args:
            json_path: Chemin vers le fichier JSON

        Returns:
            Instance de FeatureInsights
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Fichier d'analyse non trouvé: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        features = {}
        target_name = data.get("target", {}).get("name")

        # Parser chaque feature
        for feat_data in data.get("features", []):
            name = feat_data["name"]

            # Extraire les stats numériques si présentes
            num_stats = feat_data.get("numeric_stats") or {}
            cat_stats = feat_data.get("categorical_stats") or {}

            # Construire les top categories depuis cat_stats
            top_cats = []
            if cat_stats.get("top_levels"):
                top_cats = cat_stats["top_levels"]

            insight = FeatureInsight(
                name=name,
                role=feat_data.get("role", "feature"),
                inferred_type=feat_data.get("inferred_type", "unknown"),
                n_rows=feat_data.get("n_rows", 0),
                n_missing=feat_data.get("n_missing", 0),
                missing_rate=feat_data.get("missing_rate", 0.0),
                n_unique=feat_data.get("n_unique", 0),
                unique_ratio=feat_data.get("unique_ratio", 0.0),
                min_val=num_stats.get("min"),
                max_val=num_stats.get("max"),
                mean=num_stats.get("mean"),
                std=num_stats.get("std"),
                skewness=num_stats.get("skewness"),
                categories=feat_data.get("example_values", []),
                top_categories=top_cats,
                flags=feat_data.get("flags", []),
                fe_hints=feat_data.get("fe_hints", []),
                notes=feat_data.get("notes", []),
                description=feat_data.get("feature_description"),
            )
            features[name] = insight

        return cls(features=features, target_name=target_name, analyse_path=json_path)

    @classmethod
    def from_json_with_correlations(
        cls,
        stats_json_path: Union[str, Path],
        correlations_json_path: Optional[Union[str, Path]] = None,
    ) -> FeatureInsights:
        """
        Charge les insights depuis le JSON stats + corrélations optionnelles.

        Args:
            stats_json_path: Chemin vers report_stats.json
            correlations_json_path: Chemin vers le JSON des corrélations (optionnel)

        Returns:
            Instance de FeatureInsights avec corrélations
        """
        # Charger les stats de base
        insights = cls.from_json(stats_json_path)

        # Charger les corrélations si disponibles
        if correlations_json_path:
            corr_path = Path(correlations_json_path)
            if corr_path.exists():
                with open(corr_path, encoding="utf-8") as f:
                    corr_data = json.load(f)

                # Ajouter les corrélations aux features
                for feat_corr in corr_data.get("features", []):
                    name = feat_corr.get("feature")
                    if name in insights.features:
                        insights.features[name].correlation = feat_corr.get("pearson")
                        insights.features[name].correlation_spearman = feat_corr.get("spearman")
                        insights.features[name].mutual_info = feat_corr.get("mutual_info")
                        insights.features[name].combined_score = feat_corr.get("combined_score")

        return insights

    @classmethod
    def from_analyse(
        cls,
        df: pd.DataFrame,
        target_col: str,
        project_name: str,
        compute_correlations: bool = True,
    ) -> FeatureInsights:
        """
        Lance l'analyse via src/analyse/ et charge le résultat.

        Cette méthode utilise le module d'analyse existant pour garantir
        la cohérence des calculs (source unique de vérité).

        Args:
            df: DataFrame avec features et target
            target_col: Nom de la colonne cible
            project_name: Nom du projet (pour les chemins de sortie)
            compute_correlations: Calculer les corrélations avancées

        Returns:
            Instance de FeatureInsights
        """
        # Import des modules d'analyse
        import src.analyse.statistiques.report as report
        from src.analyse.path_config import AnalysePathConfig

        print("📊 Lancement de l'analyse via src/analyse/...")

        # Créer la configuration des chemins pour l'analyse
        analyse_path_config = AnalysePathConfig(project_name=project_name)

        # Lancer l'analyse statistique
        feature_cols = [c for c in df.columns if c != target_col]

        report_data = report.analyze_dataset_for_fe(
            df,
            target_cols=target_col,
            print_report=False,  # Pas d'affichage pour ne pas polluer
            dataset_name=project_name,
            business_description=f"Analyse automatique pour {project_name}",
        )

        # Sauvegarder le rapport
        stats_payload = report_data.get("llm_payload", report_data)
        json_path = analyse_path_config.save_stats_report(stats_payload)

        print(f"✅ Analyse sauvegardée: {json_path}")

        # Calculer les corrélations si demandé
        if compute_correlations:
            try:
                from src.analyse.correlation.correlation import FeatureCorrelationAnalyzer

                print("📈 Calcul des corrélations avancées...")

                # Déterminer le type de tâche
                n_unique_target = df[target_col].nunique()
                task = "classification" if n_unique_target <= 10 else "regression"

                analyzer = FeatureCorrelationAnalyzer(df, target_col=target_col, task=task)
                corr_scores = analyzer.combined_feature_score()

                # Charger les insights depuis le JSON généré
                insights = cls.from_json(json_path)

                # Ajouter les corrélations
                for _, row in corr_scores.iterrows():
                    feat_name = row["feature"]
                    if feat_name in insights.features:
                        insights.features[feat_name].correlation = row.get("pearson")
                        insights.features[feat_name].correlation_spearman = row.get("spearman")
                        insights.features[feat_name].mutual_info = row.get("mutual_info")
                        insights.features[feat_name].combined_score = row.get("combined_score")

                print(f"✅ Corrélations calculées pour {len(corr_scores)} features")
                return insights

            except ImportError as e:
                print(f"⚠️ Module de corrélation non disponible: {e}")
                print("   → Chargement sans corrélations")
            except Exception as e:
                print(f"⚠️ Erreur lors du calcul des corrélations: {e}")
                print("   → Chargement sans corrélations")

        # Charger depuis le JSON généré
        return cls.from_json(json_path)

    def get_feature(self, name: str) -> Optional[FeatureInsight]:
        """Retourne l'insight pour une feature donnée."""
        return self.features.get(name)

    def get_all_features(self) -> list[FeatureInsight]:
        """Retourne tous les insights."""
        return list(self.features.values())

    def get_features_by_importance(self) -> list[FeatureInsight]:
        """
        Retourne les features triées par importance (score combiné ou corrélation).
        """

        def get_score(f: FeatureInsight) -> float:
            if f.combined_score is not None:
                return abs(f.combined_score)
            if f.correlation is not None:
                return abs(f.correlation)
            return 0.0

        return sorted(self.features.values(), key=get_score, reverse=True)

    def get_high_value_features(self, threshold: float = 0.3) -> list[FeatureInsight]:
        """Retourne les features avec une corrélation élevée."""
        return [
            f
            for f in self.features.values()
            if f.correlation is not None and abs(f.correlation) >= threshold
        ]

    def get_low_value_features(self) -> list[FeatureInsight]:
        """Retourne les features à faible valeur (ID, constantes, etc.)."""
        return [f for f in self.features.values() if "ID_LIKE" in f.flags or "CONSTANT" in f.flags]

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "target_name": self.target_name,
            "analyse_path": str(self.analyse_path) if self.analyse_path else None,
            "features": {
                name: {
                    "name": f.name,
                    "role": f.role,
                    "inferred_type": f.inferred_type,
                    "missing_rate": f.missing_rate,
                    "n_unique": f.n_unique,
                    "correlation": f.correlation,
                    "combined_score": f.combined_score,
                    "flags": f.flags,
                    "fe_hints": f.fe_hints,
                }
                for name, f in self.features.items()
            },
        }

    def __repr__(self) -> str:
        source = f", source='{self.analyse_path}'" if self.analyse_path else ""
        return (
            f"FeatureInsights({len(self.features)} features, target='{self.target_name}'{source})"
        )
