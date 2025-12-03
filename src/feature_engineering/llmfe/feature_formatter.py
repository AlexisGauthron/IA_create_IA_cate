# src/feature_engineering/llmfe/feature_formatter.py
"""
Module pour formater les features pour le prompt LLM.
Supporte 3 formats : basic, tags, hierarchical.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
import pandas as pd

from src.feature_engineering.llmfe.feature_insights import FeatureInsights, FeatureInsight


class FeatureFormat(Enum):
    """Format de présentation des features dans le prompt."""
    BASIC = "basic"  # Format actuel : nom, type, range/catégories
    TAGS = "tags"  # Format compact avec tags inline
    HIERARCHICAL = "hierarchical"  # Format structuré par importance


class FeatureFormatter:
    """
    Formate les features pour le prompt LLM selon le format choisi.

    Exemples de sortie:

    BASIC:
        - Age: Age (numerical variable within range [0.42, 80.0])
        - Sex: Sex (categorical variable with categories [male, female])

    TAGS:
        - Age: Passenger age [NUM] [20% NaN] [CORR:-0.08] (0.42 to 80.0)
        - Sex: Gender [CAT] [HIGH_PRED] [CORR:0.54] (male, female)

    HIERARCHICAL:
        ### High-Value Features (strong predictors):
        - Sex: Gender [CORR: 0.54] (male/female)
        - Pclass: Ticket class [CORR: 0.34] (1, 2, 3)

        ### Medium-Value Features:
        - Age: Passenger age (0.42-80.0) [20% missing]

        ### Low-Value Features (consider dropping):
        - PassengerId: Identifier [NO predictive value]
    """

    def __init__(
        self,
        insights: FeatureInsights,
        format_type: FeatureFormat = FeatureFormat.BASIC,
    ):
        self.insights = insights
        self.format_type = format_type

    def format(self) -> str:
        """
        Formate toutes les features selon le format choisi.

        Returns:
            String formatée pour le prompt LLM
        """
        if self.format_type == FeatureFormat.BASIC:
            return self._format_basic()
        elif self.format_type == FeatureFormat.TAGS:
            return self._format_tags()
        elif self.format_type == FeatureFormat.HIERARCHICAL:
            return self._format_hierarchical()
        else:
            return self._format_basic()

    def _format_basic(self) -> str:
        """
        Format basique (comportement actuel).
        Juste le nom, type, et range/catégories.
        """
        lines = []
        for feat in self.insights.get_all_features():
            desc = feat.description or feat.name.replace("_", " ")

            if feat.inferred_type == "numeric":
                if feat.min_val is not None and feat.max_val is not None:
                    line = f"- {feat.name}: {desc} (numerical variable within range [{feat.min_val}, {feat.max_val}])"
                else:
                    line = f"- {feat.name}: {desc} (numerical variable)"
            elif feat.inferred_type in ("categorical_low", "categorical_high"):
                cats = ", ".join(feat.categories[:5])
                if len(feat.categories) > 5:
                    cats += ", ..."
                line = f"- {feat.name}: {desc} (categorical variable with categories [{cats}])"
            elif feat.inferred_type == "text":
                line = f"- {feat.name}: {desc} (text variable)"
            else:
                line = f"- {feat.name}: {desc} (feature)"

            lines.append(line)

        return "\n".join(lines)

    def _format_tags(self) -> str:
        """
        Format compact avec tags inline.
        Exemple: - Age: Passenger age [NUM] [20% NaN] [CORR:-0.08] (0.42 to 80.0)
        """
        lines = []
        for feat in self.insights.get_all_features():
            desc = feat.description or feat.name.replace("_", " ")

            # Tags de type
            tags = []
            if feat.inferred_type == "numeric":
                tags.append("[NUM]")
            elif feat.inferred_type in ("categorical_low", "categorical_high"):
                tags.append("[CAT]")
            elif feat.inferred_type == "text":
                tags.append("[TEXT]")
            elif feat.inferred_type == "datetime":
                tags.append("[DATE]")

            # Tag de missing
            if feat.missing_rate > 0:
                pct = int(feat.missing_rate * 100)
                tags.append(f"[{pct}% NaN]")

            # Tag de corrélation
            if feat.correlation is not None:
                corr = feat.correlation
                if abs(corr) >= 0.3:
                    tags.append(f"[HIGH_PRED]")
                tags.append(f"[CORR:{corr:.2f}]")

            # Tags de flags
            for flag in feat.flags:
                if flag == "ID_LIKE":
                    tags.append("[ID]")
                elif flag == "CONSTANT":
                    tags.append("[CONST]")
                elif flag == "HIGH_CARDINALITY":
                    tags.append("[HIGH_CARD]")
                elif flag == "SKEWED":
                    tags.append("[SKEWED]")

            # Valeurs
            if feat.inferred_type == "numeric":
                if feat.min_val is not None and feat.max_val is not None:
                    values = f"({feat.min_val:.2f} to {feat.max_val:.2f})"
                else:
                    values = ""
            elif feat.inferred_type in ("categorical_low", "categorical_high"):
                cats = ", ".join(feat.categories[:4])
                if len(feat.categories) > 4:
                    cats += f", +{len(feat.categories) - 4} more"
                values = f"({cats})"
            else:
                values = ""

            # Assembler la ligne
            tags_str = " ".join(tags)
            line = f"- {feat.name}: {desc} {tags_str} {values}".strip()
            lines.append(line)

        return "\n".join(lines)

    def _format_hierarchical(self) -> str:
        """
        Format structuré par importance.
        Groupe les features en High/Medium/Low value.
        """
        high_value = []
        medium_value = []
        low_value = []

        for feat in self.insights.get_all_features():
            # Classifier par importance
            if "ID_LIKE" in feat.flags or "CONSTANT" in feat.flags:
                low_value.append(feat)
            elif feat.correlation is not None and abs(feat.correlation) >= 0.3:
                high_value.append(feat)
            elif feat.correlation is not None and abs(feat.correlation) >= 0.1:
                medium_value.append(feat)
            elif "HIGH_CARDINALITY" in feat.flags or feat.missing_rate > 0.5:
                low_value.append(feat)
            else:
                medium_value.append(feat)

        # Trier chaque groupe par corrélation décroissante
        def sort_key(f: FeatureInsight) -> float:
            return abs(f.correlation) if f.correlation is not None else 0.0

        high_value.sort(key=sort_key, reverse=True)
        medium_value.sort(key=sort_key, reverse=True)

        lines = []

        # High-Value Features
        if high_value:
            lines.append("### High-Value Features (strong predictors):")
            for feat in high_value:
                line = self._format_hierarchical_feature(feat)
                lines.append(line)
            lines.append("")

        # Medium-Value Features
        if medium_value:
            lines.append("### Medium-Value Features:")
            for feat in medium_value:
                line = self._format_hierarchical_feature(feat)
                lines.append(line)
            lines.append("")

        # Low-Value Features
        if low_value:
            lines.append("### Low-Value Features (consider dropping):")
            for feat in low_value:
                line = self._format_hierarchical_feature(feat, show_warning=True)
                lines.append(line)

        return "\n".join(lines)

    def _format_hierarchical_feature(
        self,
        feat: FeatureInsight,
        show_warning: bool = False,
    ) -> str:
        """Formate une feature pour le format hiérarchique."""
        desc = feat.description or feat.name.replace("_", " ")

        # Infos principales
        infos = []

        # Corrélation
        if feat.correlation is not None:
            infos.append(f"CORR: {feat.correlation:.2f}")

        # Missing
        if feat.missing_rate > 0:
            pct = int(feat.missing_rate * 100)
            infos.append(f"{pct}% missing")

        # Warnings pour low-value
        if show_warning:
            if "ID_LIKE" in feat.flags:
                infos.append("ID - no predictive value")
            elif "CONSTANT" in feat.flags:
                infos.append("constant - no variance")
            elif "HIGH_CARDINALITY" in feat.flags:
                infos.append(f"high cardinality: {feat.n_unique} unique")

        # Valeurs
        if feat.inferred_type == "numeric":
            if feat.min_val is not None and feat.max_val is not None:
                values = f"({feat.min_val:.1f}-{feat.max_val:.1f})"
            else:
                values = ""
        elif feat.inferred_type in ("categorical_low", "categorical_high"):
            cats = "/".join(feat.categories[:3])
            if len(feat.categories) > 3:
                cats += "/..."
            values = f"({cats})"
        else:
            values = ""

        # Assembler
        infos_str = f"[{', '.join(infos)}]" if infos else ""
        line = f"- {feat.name}: {desc} {infos_str} {values}".strip()

        return line


def format_features_for_prompt(
    df: pd.DataFrame,
    target_col: str,
    format_type: FeatureFormat = FeatureFormat.BASIC,
    insights: Optional[FeatureInsights] = None,
    meta_data: Optional[dict] = None,
) -> str:
    """
    Fonction utilitaire pour formater les features pour un prompt.

    Args:
        df: DataFrame avec les données
        target_col: Nom de la colonne cible
        format_type: Format de sortie (BASIC, TAGS, HIERARCHICAL)
        insights: FeatureInsights pré-calculés (optionnel)
        meta_data: Descriptions des features (optionnel)

    Returns:
        String formatée pour le prompt
    """
    # Créer les insights si non fournis
    if insights is None:
        insights = FeatureInsights.from_dataframe(
            df=df,
            target_col=target_col,
            meta_data=meta_data,
            compute_correlations=True,
        )

    # Formater
    formatter = FeatureFormatter(insights=insights, format_type=format_type)
    return formatter.format()
