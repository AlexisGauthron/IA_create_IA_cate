"""
Composant de visualisation pour LLMFE (Feature Engineering).
Affiche la progression en temps réel et les résultats finaux.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time


class LLMFEProgressTracker:
    """
    Tracker de progression pour LLMFE en temps réel.

    Usage:
        tracker = LLMFEProgressTracker(max_samples=20)
        tracker.render_header()

        # Dans la boucle LLMFE:
        tracker.update(sample_num=5, score=0.85, is_valid=True, code="...")
    """

    def __init__(self, max_samples: int = 20):
        self.max_samples = max_samples
        self.samples_history: List[Dict] = []
        self.best_score = None
        self.best_sample = None

        # Placeholders Streamlit pour mise à jour dynamique
        self._header_placeholder = None
        self._progress_placeholder = None
        self._metrics_placeholder = None
        self._chart_placeholder = None
        self._log_placeholder = None

    def render_header(self):
        """Affiche l'en-tête avec les placeholders."""
        st.markdown("### 🔧 Feature Engineering en cours...")

        # Créer les placeholders
        self._progress_placeholder = st.empty()
        self._metrics_placeholder = st.empty()
        self._chart_placeholder = st.empty()

        st.markdown("#### 📝 Derniers essais")
        self._log_placeholder = st.empty()

    def update(
        self,
        sample_num: int,
        score: Optional[float],
        is_valid: bool,
        code: str = "",
        sample_time: float = 0,
        eval_time: float = 0,
    ):
        """Met à jour l'affichage avec un nouveau sample."""
        # Enregistrer le sample
        sample_data = {
            "num": sample_num,
            "score": score,
            "is_valid": is_valid,
            "code": code[:200] + "..." if len(code) > 200 else code,
            "sample_time": sample_time,
            "eval_time": eval_time,
        }
        self.samples_history.append(sample_data)

        # Mettre à jour le meilleur score
        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score = score
            self.best_sample = sample_num

        # Mettre à jour l'affichage
        self._update_progress(sample_num)
        self._update_metrics()
        self._update_chart()
        self._update_log()

    def _update_progress(self, current: int):
        """Met à jour la barre de progression."""
        if self._progress_placeholder:
            progress = min(current / self.max_samples, 1.0)
            self._progress_placeholder.progress(
                progress,
                text=f"Itération {current}/{self.max_samples}"
            )

    def _update_metrics(self):
        """Met à jour les métriques."""
        if self._metrics_placeholder:
            valid_count = sum(1 for s in self.samples_history if s["is_valid"])
            failed_count = len(self.samples_history) - valid_count

            with self._metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "🏆 Meilleur score",
                        f"{self.best_score:.4f}" if self.best_score else "—"
                    )

                with col2:
                    st.metric("✅ Valides", valid_count)

                with col3:
                    st.metric("❌ Échouées", failed_count)

                with col4:
                    success_rate = (valid_count / len(self.samples_history) * 100) if self.samples_history else 0
                    st.metric("📊 Taux succès", f"{success_rate:.0f}%")

    def _update_chart(self):
        """Met à jour le graphique d'évolution."""
        if self._chart_placeholder and self.samples_history:
            # Préparer les données pour le graphique
            valid_samples = [s for s in self.samples_history if s["score"] is not None]

            if valid_samples:
                df = pd.DataFrame(valid_samples)

                # Calculer le meilleur score cumulatif
                best_so_far = []
                current_best = float('-inf')
                for score in df["score"]:
                    current_best = max(current_best, score)
                    best_so_far.append(current_best)
                df["best_score"] = best_so_far

                with self._chart_placeholder.container():
                    st.line_chart(
                        df.set_index("num")[["score", "best_score"]],
                        use_container_width=True,
                    )

    def _update_log(self):
        """Met à jour le log des derniers essais."""
        if self._log_placeholder:
            # Afficher les 5 derniers essais
            recent = self.samples_history[-5:][::-1]

            with self._log_placeholder.container():
                for sample in recent:
                    status = "✅" if sample["is_valid"] else "❌"
                    score_str = f"{sample['score']:.4f}" if sample["score"] else "—"

                    with st.expander(
                        f"{status} Sample #{sample['num']} | Score: {score_str}",
                        expanded=(sample == recent[0])  # Expand le plus récent
                    ):
                        st.code(sample["code"], language="python")
                        st.caption(
                            f"⏱️ Sampling: {sample['sample_time']:.1f}s | "
                            f"Évaluation: {sample['eval_time']:.1f}s"
                        )


class LLMFEResultsDashboard:
    """
    Dashboard complet des résultats LLMFE.

    Usage:
        dashboard = LLMFEResultsDashboard(results_dir="/path/to/results")
        dashboard.render()
    """

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self._load_data()

    def _load_data(self):
        """Charge les données depuis les fichiers JSON."""
        self.summary = self._load_json("summary.json")
        self.best_model = self._load_json("best_model.json")
        self.all_scores = self._load_json("all_scores.json")

        # Charger les samples individuels
        self.samples = []
        samples_dir = self.results_dir.parent / "samples"
        if samples_dir.exists():
            for sample_file in sorted(samples_dir.glob("*.json")):
                try:
                    with open(sample_file) as f:
                        self.samples.append(json.load(f))
                except Exception:
                    pass

    def _load_json(self, filename: str) -> Optional[Dict]:
        """Charge un fichier JSON."""
        path = self.results_dir / filename
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def render(self):
        """Affiche le dashboard complet."""
        st.markdown("### 📊 Résultats Feature Engineering")

        # Métriques principales
        self._render_main_metrics()

        # Onglets
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Meilleur modèle",
            "📈 Évolution",
            "📋 Tous les essais",
            "📊 Statistiques"
        ])

        with tab1:
            self._render_best_model()

        with tab2:
            self._render_evolution()

        with tab3:
            self._render_all_samples()

        with tab4:
            self._render_statistics()

    def _render_main_metrics(self):
        """Affiche les métriques principales."""
        if not self.summary:
            st.warning("Résumé non disponible.")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            best_score = self.summary.get("best_score", 0)
            st.metric("🏆 Meilleur Score", f"{best_score:.4f}")

        with col2:
            total = self.summary.get("total_samples", 0)
            st.metric("🔄 Total Samples", total)

        with col3:
            valid = self.summary.get("valid_samples", 0)
            total = self.summary.get("total_samples", 1)
            rate = (valid / total * 100) if total > 0 else 0
            st.metric("✅ Taux Succès", f"{rate:.0f}%")

        with col4:
            total_time = (
                self.summary.get("total_sample_time", 0) +
                self.summary.get("total_evaluate_time", 0)
            )
            st.metric("⏱️ Temps Total", f"{total_time:.0f}s")

    def _render_best_model(self):
        """Affiche le meilleur modèle."""
        if not self.best_model:
            st.info("Aucun modèle valide généré.")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Informations")
            st.metric("Score", f"{self.best_model.get('score', 0):.4f}")
            st.metric("Sample #", self.best_model.get("sample_order", "?"))

        with col2:
            st.markdown("#### Code de transformation")
            code = self.best_model.get("function_code", "# Pas de code disponible")
            st.code(code, language="python")

            # Bouton copier
            st.download_button(
                "📋 Télécharger le code",
                data=code,
                file_name="best_features.py",
                mime="text/plain",
            )

    def _render_evolution(self):
        """Affiche l'évolution des scores."""
        if not self.all_scores:
            st.info("Données d'évolution non disponibles.")
            return

        df = pd.DataFrame(self.all_scores)

        if "sample_order" not in df.columns or "score" not in df.columns:
            st.warning("Format de données incorrect.")
            return

        df = df.sort_values("sample_order")

        # Calculer le meilleur score cumulatif
        best_so_far = []
        current_best = float('-inf')
        for score in df["score"]:
            current_best = max(current_best, score)
            best_so_far.append(current_best)
        df["Meilleur cumulé"] = best_so_far
        df = df.rename(columns={"score": "Score"})

        # Graphique
        st.markdown("#### Évolution du score")
        st.line_chart(
            df.set_index("sample_order")[["Score", "Meilleur cumulé"]],
            use_container_width=True,
        )

        # Point où le meilleur score a été atteint
        if self.best_model:
            best_sample = self.best_model.get("sample_order")
            if best_sample:
                st.success(f"🎯 Meilleur score atteint à l'itération #{best_sample}")

    def _render_all_samples(self):
        """Affiche tous les samples dans un tableau."""
        if not self.samples:
            st.info("Aucun sample disponible.")
            return

        # Créer le DataFrame
        data = []
        for sample in self.samples:
            data.append({
                "#": sample.get("sample_order", "?"),
                "Score": sample.get("score"),
                "Statut": "✅" if sample.get("score") is not None else "❌",
            })

        df = pd.DataFrame(data)
        df = df.sort_values("#", ascending=False)

        # Afficher le tableau
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Sélecteur pour voir le code
        st.markdown("#### Voir le code d'un sample")
        sample_nums = [s.get("sample_order", 0) for s in self.samples]
        selected = st.selectbox("Sélectionner un sample", sorted(sample_nums, reverse=True))

        if selected is not None:
            for sample in self.samples:
                if sample.get("sample_order") == selected:
                    st.code(sample.get("function", "# Pas de code"), language="python")
                    break

    def _render_statistics(self):
        """Affiche les statistiques détaillées."""
        if not self.summary:
            st.info("Statistiques non disponibles.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Temps d'exécution")
            sample_time = self.summary.get("total_sample_time", 0)
            eval_time = self.summary.get("total_evaluate_time", 0)

            time_df = pd.DataFrame({
                "Étape": ["Sampling (LLM)", "Évaluation"],
                "Temps (s)": [sample_time, eval_time]
            })
            st.bar_chart(time_df.set_index("Étape"))

        with col2:
            st.markdown("#### Répartition des samples")
            valid = self.summary.get("valid_samples", 0)
            failed = self.summary.get("failed_samples", 0)

            status_df = pd.DataFrame({
                "Statut": ["Valides", "Échouées"],
                "Nombre": [valid, failed]
            })
            st.bar_chart(status_df.set_index("Statut"))

        # Distribution des scores
        if self.all_scores:
            st.markdown("#### Distribution des scores")
            scores = [s["score"] for s in self.all_scores if s.get("score") is not None]

            if scores:
                score_stats = self.summary.get("score_stats", {})

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max", f"{score_stats.get('max', max(scores)):.4f}")
                with col2:
                    st.metric("Min", f"{score_stats.get('min', min(scores)):.4f}")
                with col3:
                    st.metric("Moyenne", f"{score_stats.get('mean', sum(scores)/len(scores)):.4f}")

                # Histogramme
                hist_df = pd.DataFrame({"Score": scores})
                st.bar_chart(hist_df["Score"].value_counts().sort_index())


def render_llmfe_live_progress(
    placeholder,
    sample_num: int,
    max_samples: int,
    best_score: Optional[float],
    valid_count: int,
    failed_count: int,
    last_code: str = "",
    last_score: Optional[float] = None,
):
    """
    Fonction helper pour afficher la progression LLMFE.

    Args:
        placeholder: st.empty() placeholder
        sample_num: Numéro du sample actuel
        max_samples: Nombre max de samples
        best_score: Meilleur score jusqu'ici
        valid_count: Nombre de samples valides
        failed_count: Nombre de samples échoués
        last_code: Code du dernier sample
        last_score: Score du dernier sample
    """
    with placeholder.container():
        # Progress bar
        progress = min(sample_num / max_samples, 1.0)
        st.progress(progress, text=f"🔧 Itération {sample_num}/{max_samples}")

        # Métriques
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🏆 Meilleur",
                f"{best_score:.4f}" if best_score is not None else "—"
            )

        with col2:
            st.metric("✅ Valides", valid_count)

        with col3:
            st.metric("❌ Échouées", failed_count)

        with col4:
            total = valid_count + failed_count
            rate = (valid_count / total * 100) if total > 0 else 0
            st.metric("📊 Succès", f"{rate:.0f}%")

        # Dernier sample
        if last_code:
            status = "✅" if last_score is not None else "❌"
            score_str = f"{last_score:.4f}" if last_score is not None else "Erreur"

            with st.expander(f"{status} Dernier essai | Score: {score_str}", expanded=True):
                st.code(last_code[:500] + "..." if len(last_code) > 500 else last_code, language="python")
