"""
Composants de visualisation des résultats pour le pipeline ML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


class MetricCard:
    """Affiche une métrique dans une carte stylisée."""

    @staticmethod
    def render(
        label: str,
        value: Any,
        delta: float | None = None,
        delta_color: str = "normal",
        icon: str = "",
    ):
        """
        Affiche une métrique.

        Args:
            label: Libellé de la métrique
            value: Valeur à afficher
            delta: Variation (optionnel)
            delta_color: Couleur du delta (normal, inverse, off)
            icon: Emoji/icône à afficher
        """
        col1, col2 = st.columns([1, 4])
        with col1:
            if icon:
                st.markdown(f"<span style='font-size: 2rem;'>{icon}</span>", unsafe_allow_html=True)
        with col2:
            st.metric(
                label=label,
                value=value,
                delta=delta,
                delta_color=delta_color,
            )


class AnalysisResultsComponent:
    """Composant pour afficher les résultats de l'analyse statistique."""

    def __init__(self, stats_report: dict[str, Any]):
        """
        Args:
            stats_report: Rapport statistique (llm_payload) généré par src/analyse
                Structure: {context, basic_stats, target, features (liste), analysis_config, global_notes}
        """
        self.report = stats_report
        self.basic_stats = stats_report.get("basic_stats", {})
        self.features_list = stats_report.get("features", [])
        self.target_info = stats_report.get("target", {})

    def render_summary(self):
        """Affiche le résumé de l'analyse."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_features = self.basic_stats.get("n_features", len(self.features_list))
            st.metric("Features", n_features)

        with col2:
            missing = sum(1 for feat in self.features_list if feat.get("missing_ratio", 0) > 0)
            st.metric("Avec manquants", missing)

        with col3:
            categorical = self.basic_stats.get("n_categorical_features", 0)
            st.metric("Catégorielles", categorical)

        with col4:
            numerical = self.basic_stats.get("n_numeric_features", 0)
            st.metric("Numériques", numerical)

    def render_target_info(self):
        """Affiche les informations sur la cible."""
        if not self.target_info:
            st.info("Pas d'information sur la cible.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Nom:** `{self.target_info.get('name', 'N/A')}`")

        with col2:
            task_type = self.target_info.get("task_type", "classification")
            st.markdown(f"**Type:** {task_type}")

        with col3:
            n_classes = self.target_info.get("n_classes")
            if n_classes:
                st.markdown(f"**Classes:** {n_classes}")

        # Distribution des classes si disponible
        class_distribution = self.target_info.get("class_distribution", {})
        if class_distribution:
            st.markdown("**Distribution:**")
            dist_data = [{"Classe": k, "Count": v} for k, v in class_distribution.items()]
            st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)

    def render_columns_table(self):
        """Affiche le tableau des features."""
        if not self.features_list:
            st.info("Aucune feature analysée.")
            return

        data = []
        for feat in self.features_list:
            data.append(
                {
                    "Feature": feat.get("name", "?"),
                    "Type": feat.get("inferred_type", feat.get("dtype", "?")),
                    "Uniques": feat.get("n_unique", 0),
                    "Manquants": f"{feat.get('missing_ratio', 0)*100:.1f}%",
                    "Flags": ", ".join(feat.get("flags", [])) or "-",
                }
            )

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    def render_correlations(self):
        """Affiche les corrélations avec la cible (depuis les features)."""
        # Les corrélations peuvent être dans chaque feature
        corr_data = []
        for feat in self.features_list:
            corr = feat.get("target_correlation")
            if corr is not None and isinstance(corr, (int, float)):
                corr_data.append(
                    {
                        "Feature": feat.get("name", "?"),
                        "Corrélation": corr,
                    }
                )

        if not corr_data:
            st.info("Pas de corrélations disponibles.")
            return

        # Trier par corrélation absolue
        corr_data.sort(key=lambda x: abs(x["Corrélation"]), reverse=True)

        for item in corr_data[:10]:
            corr_val = item["Corrélation"]
            item["Force"] = "🔴" if abs(corr_val) > 0.5 else "🟡" if abs(corr_val) > 0.2 else "🟢"
            item["Corrélation"] = f"{corr_val:.3f}"

        df = pd.DataFrame(corr_data[:10])
        st.dataframe(df, use_container_width=True, hide_index=True)

    def render(self):
        """Rendu complet du composant."""
        st.markdown("### 📊 Résultats de l'analyse")

        self.render_summary()

        tab1, tab2, tab3, tab4 = st.tabs(["Cible", "Features", "Corrélations", "JSON"])

        with tab1:
            self.render_target_info()

        with tab2:
            self.render_columns_table()

        with tab3:
            self.render_correlations()

        with tab4:
            st.json(self.report)


class FEResultsComponent:
    """Composant pour afficher les résultats du Feature Engineering."""

    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: Chemin vers le dossier llmfe/results
        """
        self.results_dir = Path(results_dir)

    def _load_json(self, filename: str) -> dict | None:
        """Charge un fichier JSON."""
        path = self.results_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def render_best_model(self):
        """Affiche le meilleur modèle."""
        best_model = self._load_json("best_model.json")

        if not best_model:
            st.warning("Meilleur modèle non disponible.")
            return

        col1, col2 = st.columns([1, 3])

        with col1:
            score = best_model.get("score", 0)
            st.metric("Score", f"{score:.4f}")
            st.metric("Sample #", best_model.get("sample_order", "?"))

        with col2:
            st.markdown("**Code de transformation:**")
            code = best_model.get("function_code", "N/A")
            st.code(code, language="python")

    def render_evolution(self):
        """Affiche l'évolution des scores."""
        scores = self._load_json("all_scores.json")

        if not scores:
            st.info("Données d'évolution non disponibles.")
            return

        # Créer un DataFrame pour le graphique
        df = pd.DataFrame(scores)
        if "sample_order" in df.columns and "score" in df.columns:
            df = df.sort_values("sample_order")

            st.line_chart(
                df.set_index("sample_order")["score"],
                use_container_width=True,
            )

    def render_summary(self):
        """Affiche le résumé de l'exécution."""
        summary = self._load_json("summary.json")

        if not summary:
            st.info("Résumé non disponible.")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total samples", summary.get("total_samples", 0))

        with col2:
            st.metric("Valides", summary.get("valid_samples", 0))

        with col3:
            st.metric("Échoués", summary.get("failed_samples", 0))

        with col4:
            best_score = summary.get("best_score", 0)
            st.metric("Meilleur score", f"{best_score:.4f}")

        # Temps d'exécution
        st.markdown("**Temps d'exécution:**")
        col1, col2 = st.columns(2)
        with col1:
            sample_time = summary.get("total_sample_time", 0)
            st.metric("Sampling", f"{sample_time:.1f}s")
        with col2:
            eval_time = summary.get("total_evaluate_time", 0)
            st.metric("Évaluation", f"{eval_time:.1f}s")

    def render(self):
        """Rendu complet du composant."""
        st.markdown("### 🔧 Résultats Feature Engineering")

        if not self.results_dir.exists():
            st.error(f"Dossier non trouvé: {self.results_dir}")
            return

        tab1, tab2, tab3 = st.tabs(["Meilleur modèle", "Évolution", "Résumé"])

        with tab1:
            self.render_best_model()

        with tab2:
            self.render_evolution()

        with tab3:
            self.render_summary()


class AutoMLResultsComponent:
    """Composant pour afficher les résultats AutoML."""

    def __init__(self, results: dict[str, Any]):
        """
        Args:
            results: Résultats AutoML
        """
        self.results = results

    def render_leaderboard(self):
        """Affiche le classement des modèles."""
        models = self.results.get("leaderboard", [])

        if not models:
            # Affichage basique si pas de leaderboard
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Meilleur modèle", self.results.get("best_model", "N/A"))
            with col2:
                score = self.results.get("best_score", 0)
                st.metric("Score", f"{score:.4f}" if isinstance(score, float) else score)
            with col3:
                st.metric("Framework", self.results.get("framework", "N/A"))
            return

        # Créer DataFrame du leaderboard
        df = pd.DataFrame(models)
        st.dataframe(df, use_container_width=True, hide_index=True)

    def render_feature_importance(self):
        """Affiche l'importance des features."""
        importance = self.results.get("feature_importance", {})

        if not importance:
            st.info("Importance des features non disponible.")
            return

        # Trier par importance
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        data = []
        for feature, imp in sorted_imp[:15]:
            data.append(
                {
                    "Feature": feature,
                    "Importance": imp,
                }
            )

        if data:
            df = pd.DataFrame(data)

            # Bar chart
            st.bar_chart(df.set_index("Feature")["Importance"])

    def render_predictions(self):
        """Affiche les prédictions sur le test set."""
        predictions = self.results.get("predictions")

        if predictions is None:
            st.info("Prédictions non disponibles.")
            return

        if isinstance(predictions, pd.DataFrame):
            st.dataframe(predictions.head(100), use_container_width=True)
        elif isinstance(predictions, list):
            st.dataframe(pd.DataFrame({"Prediction": predictions[:100]}))

    def render(self):
        """Rendu complet du composant."""
        st.markdown("### 🤖 Résultats AutoML")

        tab1, tab2, tab3 = st.tabs(["Classement", "Importance", "Prédictions"])

        with tab1:
            self.render_leaderboard()

        with tab2:
            self.render_feature_importance()

        with tab3:
            self.render_predictions()


class PipelineResultsDashboard:
    """
    Dashboard complet pour tous les résultats du pipeline.
    """

    def __init__(
        self,
        stats_report: dict | None = None,
        fe_results_dir: str | None = None,
        automl_results: dict | None = None,
        project_name: str = "Projet",
    ):
        self.stats_report = stats_report
        self.fe_results_dir = fe_results_dir
        self.automl_results = automl_results
        self.project_name = project_name

    def render_header(self):
        """Affiche l'en-tête du dashboard."""
        st.markdown(
            f"""
        <div style="text-align: center; padding: 1rem; background: var(--card-bg); border-radius: 12px; margin-bottom: 1rem;">
            <h2>📊 Résultats - {self.project_name}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_pipeline_status(self):
        """Affiche le statut de chaque étape."""
        col1, col2, col3 = st.columns(3)

        with col1:
            status = "✅" if self.stats_report else "⏳"
            st.markdown(f"**{status} Analyse**")

        with col2:
            status = "✅" if self.fe_results_dir else "⏳"
            st.markdown(f"**{status} Feature Engineering**")

        with col3:
            status = "✅" if self.automl_results else "⏳"
            st.markdown(f"**{status} AutoML**")

    def render(self):
        """Rendu complet du dashboard."""
        self.render_header()
        self.render_pipeline_status()

        st.markdown("---")

        # Onglets par étape
        tabs = st.tabs(["📊 Analyse", "🔧 Features", "🤖 AutoML", "📥 Export"])

        with tabs[0]:
            if self.stats_report:
                AnalysisResultsComponent(self.stats_report).render()
            else:
                st.info("Analyse non effectuée.")

        with tabs[1]:
            if self.fe_results_dir:
                FEResultsComponent(self.fe_results_dir).render()
            else:
                st.info("Feature Engineering non effectué.")

        with tabs[2]:
            if self.automl_results:
                AutoMLResultsComponent(self.automl_results).render()
            else:
                st.info("AutoML non effectué.")

        with tabs[3]:
            self.render_export_section()

    def render_export_section(self):
        """Section d'export des résultats."""
        st.markdown("### 📥 Exporter les résultats")

        col1, col2 = st.columns(2)

        with col1:
            # Export rapport JSON
            if self.stats_report:
                json_str = json.dumps(self.stats_report, indent=2, ensure_ascii=False)
                st.download_button(
                    "📄 Télécharger rapport analyse (JSON)",
                    data=json_str,
                    file_name=f"{self.project_name}_analyse.json",
                    mime="application/json",
                )

        with col2:
            # Export résultats AutoML
            if self.automl_results:
                json_str = json.dumps(self.automl_results, indent=2, default=str)
                st.download_button(
                    "🤖 Télécharger résultats AutoML (JSON)",
                    data=json_str,
                    file_name=f"{self.project_name}_automl.json",
                    mime="application/json",
                )
