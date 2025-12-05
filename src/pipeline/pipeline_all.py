# src/pipeline/pipeline_all.py
"""
Pipeline complet: Analyse -> Feature Engineering -> AutoML

Ce module orchestre les 3 étapes du pipeline ML:
1. Analyse statistique et métier du dataset (détecte task_type, metric, etc.)
2. Feature Engineering via LLMFE (utilise les params détectés)
3. Entraînement AutoML avec plusieurs frameworks

PRINCIPE: src/analyse/ est la SOURCE UNIQUE DE VÉRITÉ pour les paramètres.
Le pipeline lit le JSON généré par l'analyse pour configurer les étapes suivantes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# =============================================================================
# CONFIGURATION DES SEUILS D'INFÉRENCE
# =============================================================================
# Modifie ces valeurs pour ajuster l'auto-détection des paramètres


@dataclass
class InferenceConfig:
    """
    Configuration des seuils pour l'inférence automatique des paramètres.

    Modifie ces valeurs pour ajuster comment le pipeline choisit:
    - feature_format (basic/tags/hierarchical)
    - max_samples (nombre d'itérations LLMFE)
    - time_budget (temps AutoML en secondes)

    Exemple:
    ```python
    # Créer une config personnalisée
    config = InferenceConfig(
        max_samples_small=5,      # Moins d'itérations pour petits datasets
        time_budget_large=600,    # Plus de temps pour grands datasets
    )

    result = run_pipeline("projet", df, "target", inference_config=config)
    ```
    """

    # -------------------------------------------------------------------------
    # FEATURE FORMAT - Seuils de complexité
    # -------------------------------------------------------------------------
    # Format "basic" si n_features <= ce seuil
    format_basic_max_features: int = 5

    # Poids pour le score de complexité (utilisé pour choisir "hierarchical")
    format_complexity_many_features: float = 0.4  # n_features > 50
    format_complexity_medium_features: float = 0.2  # n_features > 20
    format_complexity_has_text: float = 0.2  # n_text > 0
    format_complexity_high_missing: float = 0.2  # missing_rate > 0.3

    # Seuil de complexité pour passer à "hierarchical"
    format_hierarchical_threshold: float = 0.5

    # -------------------------------------------------------------------------
    # MAX SAMPLES - Nombre d'itérations LLMFE
    # -------------------------------------------------------------------------
    # Seuils de n_features
    max_samples_few_features_threshold: int = 10  # Si n_features <= 10
    max_samples_many_features_threshold: int = 30  # Si n_features > 30

    # Valeurs de max_samples correspondantes
    max_samples_small: int = 10  # Pour peu de features
    max_samples_medium: int = 15  # Pour nombre moyen de features
    max_samples_large: int = 25  # Pour beaucoup de features

    # -------------------------------------------------------------------------
    # TIME BUDGET - Temps AutoML (secondes)
    # -------------------------------------------------------------------------
    # Seuils de n_rows
    time_budget_small_rows_threshold: int = 1000  # Si n_rows < 1000
    time_budget_large_rows_threshold: int = 50000  # Si n_rows > 50000

    # Valeurs de time_budget correspondantes
    time_budget_small: int = 60  # Pour petits datasets
    time_budget_medium: int = 120  # Pour datasets moyens
    time_budget_large: int = 300  # Pour grands datasets


# Config par défaut (utilisée si aucune config fournie)
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


class DetectedParams:
    """Paramètres détectés automatiquement depuis l'analyse."""

    def __init__(
        self,
        analyse_json: dict[str, Any],
        inference_config: InferenceConfig | None = None,
    ):
        """
        Extrait les paramètres depuis le JSON d'analyse.

        Args:
            analyse_json: Dictionnaire du rapport d'analyse
            inference_config: Configuration des seuils d'inférence (optionnel)
        """
        self.raw = analyse_json
        self.config = inference_config or DEFAULT_INFERENCE_CONFIG

        # Extraire les infos de la target
        target_info = analyse_json.get("target", {})
        self.target_col = target_info.get("name", "target")
        self.problem_type = target_info.get("problem_type", "binary_classification")
        self.is_imbalanced = target_info.get("is_imbalanced", False)
        self.imbalance_ratio = target_info.get("imbalance_ratio", 1.0)
        self.n_classes = target_info.get("n_unique", 2)

        # Extraire les stats de base
        basic_stats = analyse_json.get("basic_stats", {})
        self.n_rows = basic_stats.get("n_rows", 0)
        self.n_features = basic_stats.get("n_features", 0)
        self.n_numeric = basic_stats.get("n_numeric_features", 0)
        self.n_categorical = basic_stats.get("n_categorical_features", 0)
        self.n_text = basic_stats.get("n_text_features", 0)
        self.missing_rate = basic_stats.get("missing_cell_ratio", 0.0)

        # Déduire les paramètres (utilise self.config)
        self.task_type = self._infer_task_type()

        # Métrique : priorité au LLM, sinon suggestion basée sur seuils
        self.metric_source = "suggested"  # Sera mis à jour par _resolve_metric()
        self.metric_reason = ""
        self.metric = self._resolve_metric()

        # Autres paramètres
        self.feature_format = self._infer_feature_format()
        self.max_samples = self._infer_max_samples()
        self.time_budget = self._infer_time_budget()

    def _infer_task_type(self) -> Literal["classification", "regression"]:
        """Déduit le task_type depuis problem_type."""
        if "regression" in self.problem_type:
            return "regression"
        return "classification"

    def _infer_suggested_metric(self) -> tuple[str, str]:
        """
        Déduit la métrique SUGGÉRÉE basée sur les seuils statistiques.

        Returns:
            Tuple (metric, reason) - la métrique suggérée et la justification
        """
        if self.task_type == "regression":
            return "rmse", "Régression : RMSE est la métrique standard"

        # Classification
        if self.n_classes == 2:
            if self.is_imbalanced:
                return (
                    "f1",
                    f"Classification binaire déséquilibrée (ratio {self.imbalance_ratio:.2f})",
                )
            return (
                "accuracy",
                f"Classification binaire équilibrée (ratio {self.imbalance_ratio:.2f} < seuil)",
            )
        else:
            if self.is_imbalanced:
                return (
                    "f1_macro",
                    f"Classification multiclasse déséquilibrée (ratio {self.imbalance_ratio:.2f})",
                )
            return (
                "accuracy",
                f"Classification multiclasse équilibrée (ratio {self.imbalance_ratio:.2f} < seuil)",
            )

    def _get_final_metric(self) -> tuple[str | None, str | None]:
        """
        Récupère la métrique FINALE définie par le LLM (si présente).

        Returns:
            Tuple (metric, reason) ou (None, None) si pas définie par LLM
        """
        context = self.raw.get("context", {})

        # Chercher final_metric (défini par LLM)
        final_metric = context.get("final_metric")
        final_metric_reason = context.get("final_metric_reason")

        if final_metric:
            return final_metric, final_metric_reason

        # Fallback : chercher metric (ancien format LLM)
        metric = context.get("metric")
        if metric:
            return metric, "Défini par analyse LLM métier"

        return None, None

    def _resolve_metric(self) -> str:
        """
        Résout la métrique finale.

        Priorité :
        1. final_metric du LLM (si présent)
        2. suggested_metric (basé sur seuils)
        """
        # 1. Essayer d'abord la métrique du LLM
        final_metric, final_reason = self._get_final_metric()
        if final_metric:
            self.metric_source = "llm"
            self.metric_reason = final_reason
            return final_metric

        # 2. Sinon, utiliser la suggestion basée sur les seuils
        suggested, reason = self._infer_suggested_metric()
        self.metric_source = "suggested"
        self.metric_reason = reason
        return suggested

    def _infer_feature_format(self) -> Literal["basic", "tags", "hierarchical"]:
        """Déduit le format de prompt optimal pour LLMFE."""
        cfg = self.config

        # Calculer un score de complexité
        complexity = 0.0

        if self.n_features > 50:
            complexity += cfg.format_complexity_many_features
        elif self.n_features > 20:
            complexity += cfg.format_complexity_medium_features

        if self.n_text > 0:
            complexity += cfg.format_complexity_has_text

        if self.missing_rate > 0.3:
            complexity += cfg.format_complexity_high_missing

        # Décision
        if self.n_features <= cfg.format_basic_max_features:
            return "basic"
        if complexity > cfg.format_hierarchical_threshold:
            return "hierarchical"
        return "tags"

    def _infer_max_samples(self) -> int:
        """Déduit le nombre optimal d'itérations LLMFE."""
        cfg = self.config

        if self.n_features <= cfg.max_samples_few_features_threshold:
            return cfg.max_samples_small
        if self.n_features > cfg.max_samples_many_features_threshold:
            return cfg.max_samples_large
        return cfg.max_samples_medium

    def _infer_time_budget(self) -> int:
        """Déduit le time budget optimal pour AutoML."""
        cfg = self.config

        if self.n_rows < cfg.time_budget_small_rows_threshold:
            return cfg.time_budget_small
        if self.n_rows > cfg.time_budget_large_rows_threshold:
            return cfg.time_budget_large
        return cfg.time_budget_medium

    def summary(self) -> str:
        """Résumé des paramètres détectés."""
        metric_info = f"{self.metric} ({self.metric_source})"
        return f"""
                    Paramètres détectés depuis l'analyse:
                    ─────────────────────────────────────
                    Target:         {self.target_col}
                    Problem type:   {self.problem_type}
                    Task type:      {self.task_type}
                    Metric:         {metric_info}
                    Metric reason:  {self.metric_reason}
                    Imbalanced:     {self.is_imbalanced} (ratio: {self.imbalance_ratio:.2f})

                    Dataset:        {self.n_rows} rows, {self.n_features} features
                    Feature format: {self.feature_format}
                    Max samples:    {self.max_samples}
                    Time budget:    {self.time_budget}s
                """


class PipelineResult:
    """Résultat d'exécution du pipeline complet."""

    def __init__(self):
        self.analyse_result: dict[str, Any] | None = None
        self.detected_params: DetectedParams | None = None
        self.feature_engineering_result: dict[str, Any] | None = None
        self.automl_result: dict[str, Any] | None = None

        self.df_train_fe: pd.DataFrame | None = None
        self.df_test_fe: pd.DataFrame | None = None

        self.best_model: Any | None = None
        self.best_score: float | None = None
        self.best_framework: str | None = None

        self.output_dir: Path | None = None
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def summary(self) -> dict[str, Any]:
        """Retourne un résumé des résultats."""
        return {
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "analyse_completed": self.analyse_result is not None,
            "detected_params": {
                "task_type": self.detected_params.task_type if self.detected_params else None,
                "metric": self.detected_params.metric if self.detected_params else None,
            },
            "fe_completed": self.feature_engineering_result is not None,
            "automl_completed": self.automl_result is not None,
            "best_framework": self.best_framework,
            "best_score": self.best_score,
        }


class FullPipeline:
    """
    Pipeline complet: Analyse -> Feature Engineering -> AutoML

    Les paramètres sont détectés AUTOMATIQUEMENT depuis l'analyse.
    Seuls target_col et project_name sont requis.

    Exemple d'utilisation:
    ```python
    # Usage minimal - tout est auto-détecté
    result = run_pipeline(
        project_name="titanic",
        df_train=df,
        target_col="Survived",
    )

    # Avec override manuel si besoin
    result = run_pipeline(
        project_name="titanic",
        df_train=df,
        target_col="Survived",
        override_metric="f1",  # Force F1 au lieu de l'auto-détection
    )
    ```
    """

    def __init__(
        self,
        project_name: str,
        target_col: str,
        output_dir: str = "outputs",
        # Options pour activer/désactiver les étapes
        enable_analyse: bool = True,
        enable_fe: bool = True,
        enable_automl: bool = True,
        # Force la regénération de l'analyse même si le JSON existe
        force_analyse: bool = False,
        # Overrides manuels (si None, utilise l'auto-détection)
        override_task_type: str | None = None,
        override_metric: str | None = None,
        override_feature_format: str | None = None,
        override_max_samples: int | None = None,
        override_time_budget: int | None = None,
        # Config d'inférence (seuils pour auto-détection)
        inference_config: InferenceConfig | None = None,
        # Config analyse
        analyse_only_stats: bool = True,
        analyse_provider: str = "openai",
        analyse_model: str = "gpt-4o-mini",
        # Config corrélations
        with_correlations: bool = False,
        correlation_methods: list | None = None,
        # Config LLMFE
        llmfe_model: str = "gpt-3.5-turbo",
        # Config évaluation LLMFE (multi-modèle)
        eval_metric: str = "auto",
        eval_models: list | None = None,
        eval_aggregation: str = "mean",
        # Config AutoML
        automl_frameworks: list | None = None,
    ):
        """
        Initialise le pipeline.

        Args:
            project_name: Nom du projet
            target_col: Colonne cible (OBLIGATOIRE)
            output_dir: Dossier de sortie
            enable_*: Active/désactive les étapes
            force_analyse: Force la regénération de l'analyse même si elle existe
            override_*: Force un paramètre au lieu de l'auto-détection
            inference_config: Configuration des seuils d'inférence (voir InferenceConfig)
        """
        self.project_name = project_name
        self.target_col = target_col
        self.output_dir_base = output_dir

        # Options d'étapes
        self.enable_analyse = enable_analyse
        self.enable_fe = enable_fe
        self.enable_automl = enable_automl
        self.force_analyse = force_analyse

        # Overrides
        self.override_task_type = override_task_type
        self.override_metric = override_metric
        self.override_feature_format = override_feature_format
        self.override_max_samples = override_max_samples
        self.override_time_budget = override_time_budget

        # Config d'inférence
        self.inference_config = inference_config or DEFAULT_INFERENCE_CONFIG

        # Config analyse
        self.analyse_only_stats = analyse_only_stats
        self.analyse_provider = analyse_provider
        self.analyse_model = analyse_model

        # Config corrélations
        self.with_correlations = with_correlations
        self.correlation_methods = correlation_methods

        # Config LLMFE
        self.llmfe_model = llmfe_model

        # Config évaluation LLMFE (multi-modèle)
        self.eval_metric = eval_metric
        self.eval_models = eval_models or ["xgboost"]
        self.eval_aggregation = eval_aggregation

        # Config AutoML
        self.automl_frameworks = automl_frameworks or ["flaml", "autogluon"]

        # Runtime
        # Timestamp pour les métadonnées uniquement (pas dans la structure des dossiers)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result = PipelineResult()
        self.detected_params: DetectedParams | None = None

        # Créer le dossier de sortie (sans timestamp dans le chemin)
        self.output_dir = Path(output_dir) / project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result.output_dir = self.output_dir

    def run(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
    ) -> PipelineResult:
        """
        Exécute le pipeline complet.

        Ordre:
        1. Analyse -> génère JSON avec task_type, metric, etc.
        2. Lecture du JSON -> DetectedParams
        3. Feature Engineering -> utilise DetectedParams
        4. AutoML -> utilise DetectedParams

        Args:
            df_train: DataFrame d'entraînement
            df_test: DataFrame de test (optionnel)

        Returns:
            PipelineResult avec tous les résultats
        """
        print(self._header())

        # Étape 1: Récupérer ou générer l'analyse
        analyse_json_path = self._get_or_run_analyse(df_train)

        # Étape 2: Charger les paramètres détectés
        self._load_detected_params(analyse_json_path)
        print(self.detected_params.summary())

        # Étape 3: Feature Engineering
        if self.enable_fe:
            df_train, df_test = self._run_feature_engineering(df_train, df_test)

        # Étape 4: AutoML
        if self.enable_automl:
            self._run_automl(df_train, df_test)

        # Sauvegarder le résumé
        self._save_summary()

        print(self._footer())
        return self.result

    def _get_or_run_analyse(self, df: pd.DataFrame) -> Path:
        """
        Récupère l'analyse existante ou la génère si nécessaire.

        Logique:
        1. Si force_analyse=True -> toujours regénérer
        2. Si le JSON stats existe -> le réutiliser
        3. Sinon -> générer l'analyse (avec ou sans LLM selon analyse_only_stats)

        Returns:
            Chemin vers le JSON d'analyse (existant ou nouvellement généré)
        """
        from src.analyse.path_config import AnalysePathConfig

        # Créer le path_config pour vérifier les chemins
        analyse_path_config = AnalysePathConfig(
            project_name=self.project_name,
            base_dir=self.output_dir_base,
        )

        # Vérifier si le rapport stats existe déjà
        json_path = analyse_path_config.stats_report_path

        if json_path.exists() and not self.force_analyse:
            print("\n" + "=" * 60)
            print("  ÉTAPE 1: ANALYSE DU DATASET (EXISTANTE)")
            print("=" * 60)
            print("\n  Analyse existante trouvée:")
            print(f"    {json_path}")
            print("\n  Utilisation du JSON existant (--force-analyse pour regénérer)")

            # Charger pour afficher quelques infos
            with open(json_path, encoding="utf-8") as f:
                existing_data = json.load(f)

            # Afficher un résumé
            context = existing_data.get("context", {})
            features = existing_data.get("features", [])
            print("\n  Résumé de l'analyse existante:")
            print(f"    - Dataset: {context.get('dataset_name', 'N/A')}")
            print(f"    - Target: {context.get('target_column', 'N/A')}")
            print(f"    - Features: {len(features)}")
            print(f"    - Task type: {context.get('task_type', 'N/A')}")

            self.result.analyse_result = {
                "json_path": str(json_path),
                "source": "existing",
                "n_features": len(features),
            }

            # Vérifier si le LLM est demandé mais pas encore fait
            if not self.analyse_only_stats:  # --with-llm demandé
                if not analyse_path_config.has_llm_analysis():
                    print("\n  [INFO] Analyse LLM non trouvée, lancement...")
                    self._run_llm_analysis(
                        df=None,  # Pas besoin du df si on a le JSON existant
                        analyse_path_config=analyse_path_config,
                        existing_report=existing_data,
                    )
                else:
                    print("\n  [INFO] Analyse LLM déjà existante.")

            return json_path

        # Sinon, lancer l'analyse
        if self.force_analyse and json_path.exists():
            print("\n  [INFO] --force-analyse activé, regénération de l'analyse...")

        return self._run_analyse(df)

    def _run_analyse(self, df: pd.DataFrame) -> Path:
        """
        Exécute l'étape d'analyse (génère le JSON).

        Returns:
            Chemin vers le JSON généré
        """
        print("\n" + "=" * 60)
        print("  ÉTAPE 1: ANALYSE DU DATASET (GÉNÉRATION)")
        print("=" * 60)

        import src.analyse.statistiques.report as report
        from src.analyse.path_config import AnalysePathConfig

        # Créer la config de chemins pour l'analyse
        # Note: base_dir doit être "outputs" (pas "outputs/project"), sinon duplication
        analyse_path_config = AnalysePathConfig(
            project_name=self.project_name,
            base_dir=self.output_dir_base,
        )

        print(f"  Target: {self.target_col}")
        print(f"  Lignes: {len(df)}")
        print(f"  Colonnes: {len(df.columns)}")

        # Lancer l'analyse statistique
        report_data = report.analyze_dataset_for_fe(
            df,
            target_cols=self.target_col,
            print_report=True,
            dataset_name=self.project_name,
            business_description=f"Analyse pour prédire {self.target_col}",
            # Options corrélations
            with_correlations=self.with_correlations,
            correlation_methods=self.correlation_methods,
        )

        # Sauvegarder le rapport
        stats_payload = report_data.get("llm_payload", report_data)
        json_path = analyse_path_config.save_stats_report(stats_payload)
        print(f"\n  Rapport sauvegardé: {json_path}")

        self.result.analyse_result = {
            "json_path": str(json_path),
            "n_features": len(df.columns) - 1,
            "n_rows": len(df),
        }

        # Si analyse LLM demandée (pas seulement stats)
        if not self.analyse_only_stats:
            self._run_llm_analysis(
                df=df,
                analyse_path_config=analyse_path_config,
                existing_report=stats_payload,
            )

        print("\n  Analyse terminée")
        return json_path

    def _run_llm_analysis(
        self,
        df: pd.DataFrame | None,
        analyse_path_config,
        existing_report: dict[str, Any],
    ) -> None:
        """
        Exécute l'analyse métier LLM et sauvegarde la conversation.

        Args:
            df: DataFrame (peut être None si on utilise un rapport existant)
            analyse_path_config: Configuration des chemins d'analyse
            existing_report: Rapport statistique existant (llm_payload)
        """
        print("\n  Lancement de l'analyse métier LLM...")

        try:
            from src.analyse.metier.business_agent import _is_final_mode, _process_user_input
            from src.analyse.metier.chatbot_llm import BusinessClarificationBot
            from src.core.llm_client import OllamaClient

            # Créer le client LLM
            llm_client = OllamaClient(
                provider=self.analyse_provider,
                model=self.analyse_model,
            )

            # Créer le chatbot avec le snapshot
            bot = BusinessClarificationBot(
                stats=existing_report.get("llm_payload", existing_report),
                llm=llm_client,
            )

            # Afficher les commandes disponibles
            print("\n  ╔═══════════════════════════════════════════════════════════════╗")
            print("  ║  Commandes disponibles :                                      ║")
            print("  ║  • Passer la question : skip, passe, suivant, next            ║")
            print("  ║  • Terminer et générer le rapport : done, fin, stop, terminer ║")
            print("  ║  • Interrompre : Ctrl+C                                       ║")
            print("  ╚═══════════════════════════════════════════════════════════════╝")

            # Première question du bot
            question = bot.ask_next(user_answer=None)
            print(f"\n  Agent: {question}")

            # Vérifier si la première réponse est déjà finale
            if _is_final_mode(question):
                print("\n  [Le LLM a généré le rapport final]")
            else:
                # Boucle de conversation interactive
                while True:
                    try:
                        user_input = input("\n  Vous: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n  [Conversation interrompue]")
                        break

                    if not user_input:
                        continue

                    # Formater la réponse (gère les commandes SKIP/DONE)
                    formatted_input = _process_user_input(user_input, verbose=True)

                    # Obtenir la prochaine question
                    question = bot.ask_next(user_answer=formatted_input)
                    print(f"\n  Agent: {question}")

                    # Vérifier si le LLM a terminé (Mode: Final)
                    if _is_final_mode(question):
                        print("\n  [Le LLM a généré le rapport final]")
                        break

            # 1. Trouver la réponse finale du LLM (Mode: Final)
            final_llm_report = None
            for exchange in bot.conversation_history:
                if exchange["role"] == "assistant":
                    try:
                        parsed = json.loads(exchange["content"])
                        if parsed.get("Mode") == "Final":
                            final_llm_report = parsed
                    except json.JSONDecodeError:
                        continue

            # Exporter et sauvegarder la conversation
            conversation_data = bot.export_conversation(
                provider=self.analyse_provider,
                model=self.analyse_model,
                project=self.project_name,
                final_report=final_llm_report,
            )

            conversation_path = analyse_path_config.save_conversation(conversation_data)
            print(f"\n  Conversation sauvegardée: {conversation_path}")

            # 2. Générer le rapport complet (full) avec fusion des annotations LLM
            full_report = existing_report.copy()
            full_report["llm_conversation"] = {
                "provider": self.analyse_provider,
                "model": self.analyse_model,
                "n_exchanges": len(bot.conversation_history),
            }

            # 3. Fusionner les annotations LLM dans le rapport
            if final_llm_report:
                print("\n  [INFO] Fusion des annotations LLM dans le rapport...")

                # Enrichir context
                llm_context = final_llm_report.get("context", {})
                if "context" not in full_report:
                    full_report["context"] = {}

                # Extraire les valeurs (format: {"value": "...", "confidence": 0.9})
                if llm_context.get("business_description"):
                    full_report["context"]["business_description"] = llm_context[
                        "business_description"
                    ].get("value")

                # NOUVEAU FORMAT : eval_metrics (liste de métriques avec poids)
                if llm_context.get("eval_metrics"):
                    eval_metrics_data = llm_context["eval_metrics"].get("value", [])
                    full_report["context"]["eval_metrics"] = eval_metrics_data
                    # Extraire aussi la métrique principale (celle avec le poids le plus élevé)
                    # pour rétrocompatibilité avec final_metric
                    if eval_metrics_data:
                        main_metric = max(eval_metrics_data, key=lambda m: m.get("weight", 0))
                        full_report["context"]["final_metric"] = main_metric.get("name")

                # ANCIEN FORMAT : final_metric (rétrocompatibilité)
                elif llm_context.get("final_metric"):
                    final_metric = llm_context["final_metric"].get("value")
                    full_report["context"]["final_metric"] = final_metric
                    # Convertir en nouveau format eval_metrics pour uniformité
                    full_report["context"]["eval_metrics"] = [
                        {"name": final_metric, "weight": 1.0, "reason": "Métrique unique"}
                    ]

                # Raison des métriques
                if llm_context.get("eval_metrics_reason"):
                    full_report["context"]["eval_metrics_reason"] = llm_context[
                        "eval_metrics_reason"
                    ].get("value")
                elif llm_context.get("final_metric_reason"):
                    full_report["context"]["eval_metrics_reason"] = llm_context[
                        "final_metric_reason"
                    ].get("value")

                # Enrichir features
                llm_features = {f["name"]: f for f in final_llm_report.get("features", [])}
                for feature in full_report.get("features", []):
                    if feature["name"] in llm_features:
                        llm_feat = llm_features[feature["name"]]
                        if llm_feat.get("feature_description"):
                            feature["feature_description"] = llm_feat["feature_description"].get(
                                "value"
                            )

                # Affichage
                print(
                    f"    - business_description: {full_report['context'].get('business_description', 'N/A')[:50]}..."
                )
                eval_metrics = full_report["context"].get("eval_metrics", [])
                if eval_metrics:
                    metrics_str = ", ".join(
                        f"{m['name']}={m.get('weight', 1.0)}" for m in eval_metrics
                    )
                    print(f"    - eval_metrics: [{metrics_str}]")
                else:
                    print(
                        f"    - final_metric: {full_report['context'].get('final_metric', 'N/A')}"
                    )
                print(f"    - Features enrichies: {len(llm_features)}")

                # Sauvegarder le rapport full SEULEMENT si Mode Final trouvé
                full_report_path = analyse_path_config.save_full_report(full_report)
                print(f"  Rapport complet sauvegardé: {full_report_path}")
            else:
                print("\n  [WARN] Aucune réponse finale (Mode: Final) trouvée dans la conversation")
                print("  [INFO] Le dossier full/ ne sera PAS créé (pas de métadonnées LLM fiables)")

        except Exception as e:
            print(f"\n  [ERREUR] Analyse métier LLM échouée: {e}")
            import traceback

            traceback.print_exc()

    def _normalize_metric_name(self, metric: str) -> str:
        """
        Normalise le nom d'une métrique vers le format attendu par LLMFE.

        Args:
            metric: Nom de métrique brut (ex: "ROC_AUC", "f1-macro", etc.)

        Returns:
            Nom normalisé (ex: "auc", "f1_macro", etc.)
        """
        if not metric:
            return "auto"

        metric = metric.lower().strip()

        # Mapping des alias vers les noms standardisés
        metric_aliases = {
            # AUC variants
            "roc_auc": "auc",
            "rocauc": "auc",
            "roc-auc": "auc",
            "auroc": "auc",
            # F1 variants
            "f1-score": "f1",
            "f1score": "f1",
            "f1-macro": "f1_macro",
            "f1-micro": "f1_micro",
            "f1_weighted": "f1",
            # Régression
            "mean_squared_error": "mse",
            "root_mean_squared_error": "rmse",
            "mean_absolute_error": "mae",
            "r_squared": "r2",
            "r-squared": "r2",
        }

        return metric_aliases.get(metric, metric)

    def _load_detected_params(self, json_path: Path) -> None:
        """Charge les paramètres depuis le JSON d'analyse."""
        print("\n" + "-" * 40)
        print("  Chargement des paramètres détectés...")

        with open(json_path, encoding="utf-8") as f:
            analyse_json = json.load(f)

        self.detected_params = DetectedParams(analyse_json, self.inference_config)
        self.result.detected_params = self.detected_params

        # Appliquer les overrides si spécifiés
        if self.override_task_type:
            print(f"  [Override] task_type: {self.override_task_type}")
            self.detected_params.task_type = self.override_task_type

        if self.override_metric:
            print(f"  [Override] metric: {self.override_metric}")
            self.detected_params.metric = self.override_metric

        if self.override_feature_format:
            print(f"  [Override] feature_format: {self.override_feature_format}")
            self.detected_params.feature_format = self.override_feature_format

        if self.override_max_samples:
            print(f"  [Override] max_samples: {self.override_max_samples}")
            self.detected_params.max_samples = self.override_max_samples

        if self.override_time_budget:
            print(f"  [Override] time_budget: {self.override_time_budget}")
            self.detected_params.time_budget = self.override_time_budget

    def _run_feature_engineering(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Exécute l'étape de Feature Engineering avec les params détectés."""
        print("\n" + "=" * 60)
        print("  ÉTAPE 2: FEATURE ENGINEERING (LLMFE)")
        print("=" * 60)

        from src.analyse.path_config import AnalysePathConfig
        from src.feature_engineering.llmfe.feature_formatter import FeatureFormat
        from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner
        from src.feature_engineering.path_config import FeatureEngineeringPathConfig

        params = self.detected_params

        # Créer le path_config pour FE
        # Note: base_dir doit être "outputs" (pas "outputs/project"), sinon duplication
        fe_path_config = FeatureEngineeringPathConfig(
            project_name=self.project_name,
            base_dir=self.output_dir_base,
        )

        # Convertir le format string en enum
        format_map = {
            "basic": FeatureFormat.BASIC,
            "tags": FeatureFormat.TAGS,
            "hierarchical": FeatureFormat.HIERARCHICAL,
        }
        feature_format = format_map.get(params.feature_format, FeatureFormat.TAGS)

        print(f"  Task type:      {params.task_type} (auto-détecté)")
        print(f"  Feature format: {params.feature_format} (auto-détecté)")
        print(f"  Max samples:    {params.max_samples} (auto-détecté)")
        print(f"  Modèle LLM:     {self.llmfe_model}")

        # =====================================================================
        # NOUVEAU : Charger les descriptions de l'agent LLM business si disponibles
        # =====================================================================
        analyse_path_config = AnalysePathConfig(
            project_name=self.project_name,
            base_dir=self.output_dir_base,
        )

        analyse_path = None
        task_description = None
        llm_eval_metrics = None  # Métriques pondérées recommandées par le LLM business

        # Vérifier si le rapport enrichi par le LLM business existe
        if analyse_path_config.full_report_path.exists():
            analyse_path = str(analyse_path_config.full_report_path)
            print(f"\n  📚 Rapport LLM business trouvé: {analyse_path}")

            # Charger les descriptions du LLM
            try:
                with open(analyse_path_config.full_report_path, encoding="utf-8") as f:
                    full_report = json.load(f)

                # Extraire la description métier
                context = full_report.get("context", {})
                business_desc = context.get("business_description")
                if business_desc:
                    task_description = business_desc
                    print(f"  ✅ Description métier chargée: {task_description[:60]}...")

                # NOUVEAU : Extraire les métriques pondérées (eval_metrics)
                eval_metrics = context.get("eval_metrics", [])
                if eval_metrics:
                    # Normaliser les noms des métriques
                    llm_eval_metrics = []
                    for m in eval_metrics:
                        normalized_name = self._normalize_metric_name(m.get("name", ""))
                        if normalized_name and normalized_name != "auto":
                            llm_eval_metrics.append(
                                {
                                    "name": normalized_name,
                                    "weight": m.get("weight", 1.0),
                                    "reason": m.get("reason", ""),
                                }
                            )

                    if llm_eval_metrics:
                        metrics_str = ", ".join(
                            f"{m['name']}={m['weight']}" for m in llm_eval_metrics
                        )
                        print(f"  ✅ Métriques LLM pondérées: [{metrics_str}]")

                        metric_reason = context.get("eval_metrics_reason", "")
                        if metric_reason:
                            print(f"     Raison: {metric_reason[:80]}...")

                # RÉTROCOMPATIBILITÉ : Si pas d'eval_metrics, chercher final_metric
                if not llm_eval_metrics:
                    final_metric = context.get("final_metric")
                    if final_metric:
                        normalized = self._normalize_metric_name(final_metric)
                        llm_eval_metrics = [{"name": normalized, "weight": 1.0}]
                        print(f"  ✅ Métrique LLM (unique): {normalized}")

                # Compter les features avec descriptions
                features_with_desc = sum(
                    1 for f in full_report.get("features", []) if f.get("feature_description")
                )
                if features_with_desc > 0:
                    print(f"  ✅ {features_with_desc} features avec descriptions LLM")

            except (json.JSONDecodeError, OSError) as e:
                print(f"  ⚠️ Erreur lecture rapport LLM: {e}")

        elif analyse_path_config.stats_report_path.exists():
            # Fallback sur le rapport stats (sans descriptions LLM)
            analyse_path = str(analyse_path_config.stats_report_path)
            print(f"\n  📊 Rapport stats utilisé: {analyse_path}")
            print("  ℹ️  Pas de descriptions LLM (utilisez --with-llm pour les activer)")

        # Fallback pour task_description si non définie
        if not task_description:
            task_description = f"Predict '{self.target_col}' from the given features."

        # =====================================================================

        # Créer et lancer le runner
        runner = LLMFERunner(
            project_name=self.project_name,
            path_config=fe_path_config,
        )

        is_regression = params.task_type == "regression"

        # Déterminer les métriques à utiliser :
        # 1. Si llm_eval_metrics disponible (du rapport LLM business) → l'utiliser
        # 2. Sinon utiliser self.eval_metric (défaut ou override CLI)
        if llm_eval_metrics:
            print("  📊 Utilisation des métriques LLM pondérées")
            result = runner.run(
                df_train=df_train,
                target_col=self.target_col,
                is_regression=is_regression,
                max_samples=params.max_samples,
                use_api=True,
                api_model=self.llmfe_model,
                feature_format=feature_format,
                # NOUVEAU : Paramètres d'évaluation multi-métrique pondérée
                eval_metrics_config=llm_eval_metrics,
                eval_models=self.eval_models,
                eval_aggregation=self.eval_aggregation,
                # Descriptions de l'agent LLM business
                analyse_path=analyse_path,
                task_description=task_description,
            )
        else:
            # Fallback sur métrique unique
            result = runner.run(
                df_train=df_train,
                target_col=self.target_col,
                is_regression=is_regression,
                max_samples=params.max_samples,
                use_api=True,
                api_model=self.llmfe_model,
                feature_format=feature_format,
                # Paramètres d'évaluation mono-métrique
                eval_metric=self.eval_metric,
                eval_models=self.eval_models,
                eval_aggregation=self.eval_aggregation,
                # Descriptions de l'agent LLM business
                analyse_path=analyse_path,
                task_description=task_description,
            )

        self.result.feature_engineering_result = result

        # Charger les données transformées si disponibles
        features_dir = fe_path_config.features_dir
        train_fe_path = features_dir / "train_fe.parquet"
        test_fe_path = features_dir / "test_fe.parquet"

        if train_fe_path.exists():
            df_train = pd.read_parquet(train_fe_path)
            self.result.df_train_fe = df_train
            print(f"\n  Train FE chargé: {train_fe_path}")

        if test_fe_path.exists() and df_test is not None:
            df_test = pd.read_parquet(test_fe_path)
            self.result.df_test_fe = df_test
            print(f"  Test FE chargé: {test_fe_path}")

        print("\n  Feature Engineering terminé")
        return df_train, df_test

    def _run_automl(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Exécute l'étape AutoML avec les params détectés."""
        print("\n" + "=" * 60)
        print("  ÉTAPE 3: AUTOML")
        print("=" * 60)

        from src.automl.runner import AutoMLRunner

        params = self.detected_params

        print(f"  Metric:       {params.metric} (auto-détecté)")
        print(f"  Time budget:  {params.time_budget}s (auto-détecté)")
        print(f"  Frameworks:   {', '.join(self.automl_frameworks)}")

        # Préparer les données
        X_train = df_train.drop(columns=[self.target_col])
        y_train = df_train[self.target_col]

        X_test = None
        if df_test is not None:
            if self.target_col in df_test.columns:
                X_test = df_test.drop(columns=[self.target_col])
            else:
                X_test = df_test

        # Dossier de sortie pour les modèles
        models_dir = self.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Lancer AutoML
        automl_runner = AutoMLRunner(
            output_dir=str(models_dir),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
        )

        automl_runner.use_all(model=self.automl_frameworks)

        # Collecter les résultats
        scores = {}
        if "flaml" in self.automl_frameworks and automl_runner.score_flaml is not None:
            scores["flaml"] = automl_runner.score_flaml
        if "autogluon" in self.automl_frameworks and automl_runner.score_autogluon is not None:
            scores["autogluon"] = automl_runner.score_autogluon
        if "tpot" in self.automl_frameworks and automl_runner.score_tpot is not None:
            scores["tpot"] = automl_runner.score_tpot
        if "h2o" in self.automl_frameworks and automl_runner.score_h2o is not None:
            scores["h2o"] = automl_runner.score_h2o

        # Trouver le meilleur
        if scores:
            best_framework = max(scores, key=scores.get)
            best_score = scores[best_framework]
            self.result.best_framework = best_framework
            self.result.best_score = best_score
            print(f"\n  Meilleur: {best_framework} (score: {best_score:.4f})")

        self.result.automl_result = {
            "metric_used": params.metric,
            "scores": scores,
            "best_framework": self.result.best_framework,
            "best_score": self.result.best_score,
            "models_dir": str(models_dir),
            "errors": getattr(automl_runner, "errors", {}),
        }

        print("\n  AutoML terminé")
        return self.result.automl_result

    def _save_summary(self) -> None:
        """Sauvegarde le résumé du pipeline."""
        summary = {
            "project_name": self.project_name,
            "target_col": self.target_col,
            "timestamp": self.timestamp,
            "detected_params": {
                "task_type": self.detected_params.task_type,
                "metric": self.detected_params.metric,
                "feature_format": self.detected_params.feature_format,
                "problem_type": self.detected_params.problem_type,
                "is_imbalanced": self.detected_params.is_imbalanced,
            },
            "results": self.result.summary(),
        }

        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n  Résumé sauvegardé: {summary_path}")

    def _header(self) -> str:
        """Génère le header du pipeline."""
        steps = ["Analyse"]
        if self.enable_fe:
            steps.append("FE(llmfe)")
        if self.enable_automl:
            steps.append(f"AutoML({','.join(self.automl_frameworks)})")

        return f"""
{'#' * 70}
#  FULL PIPELINE - {self.project_name.upper()}
{'#' * 70}
#
#  Timestamp: {self.timestamp}
#  Output:    {self.output_dir}
#  Target:    {self.target_col}
#  Steps:     {' -> '.join(steps)}
#
#  Note: task_type, metric, feature_format seront AUTO-DÉTECTÉS
#
{'#' * 70}
"""

    def _footer(self) -> str:
        """Génère le footer du pipeline."""
        return f"""
{'#' * 70}
#  PIPELINE TERMINÉ
{'#' * 70}
#
#  Résultats dans: {self.output_dir}
#  Task type:      {self.detected_params.task_type if self.detected_params else 'N/A'}
#  Metric:         {self.detected_params.metric if self.detected_params else 'N/A'}
#  Best framework: {self.result.best_framework or 'N/A'}
#  Best score:     {self.result.best_score or 'N/A'}
#
{'#' * 70}
"""


def run_pipeline(
    project_name: str,
    df_train: pd.DataFrame,
    target_col: str,
    df_test: pd.DataFrame | None = None,
    **kwargs,
) -> PipelineResult:
    """
    Fonction raccourcie pour lancer le pipeline complet.

    Tous les paramètres (task_type, metric, feature_format) sont
    AUTOMATIQUEMENT détectés depuis l'analyse du dataset.

    Args:
        project_name: Nom du projet
        df_train: DataFrame d'entraînement
        target_col: Colonne cible (OBLIGATOIRE)
        df_test: DataFrame de test (optionnel)
        **kwargs: Overrides optionnels (override_metric, etc.)

    Returns:
        PipelineResult avec tous les résultats

    Exemple:
    ```python
    # Usage minimal - tout est auto-détecté
    result = run_pipeline(
        project_name="titanic",
        df_train=df,
        target_col="Survived",
    )

    print(f"Task détecté: {result.detected_params.task_type}")
    print(f"Metric utilisée: {result.detected_params.metric}")
    print(f"Best score: {result.best_score}")
    ```
    """
    pipeline = FullPipeline(
        project_name=project_name,
        target_col=target_col,
        **kwargs,
    )

    return pipeline.run(df_train, df_test)
