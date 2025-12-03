from __future__ import annotations

from typing import Any, Dict, Union, Sequence, Optional, List
import pandas as pd

from .config import FEAnalysisConfig
from .targets import analyze_targets
from .features import analyze_features
from .leakage import detect_leakage
from .printing import print_fe_report

from src.analyse.dataset.all import (
    DatasetContextForLLM,
    BasicDatasetStats,
    TargetSummaryForLLM,
    FeatureSummaryForLLM,
    FEDatasetSnapshotForLLM,
)


def _compute_correlations(
    df: pd.DataFrame,
    target_col: str,
    task: str = "classification",
    methods: List[str] = None,
) -> Dict[str, Any]:
    """
    Calcule les corrélations entre les features et la cible.

    Args:
        df: DataFrame avec les données
        target_col: Nom de la colonne cible
        task: "classification" ou "regression"
        methods: Liste des méthodes à utiliser (par défaut: toutes)

    Returns:
        Dictionnaire avec les résultats de corrélation
    """
    try:
        from src.analyse.correlation.correlation import FeatureCorrelationAnalyzer
    except ImportError as e:
        print(f"[WARN] Module de corrélation non disponible: {e}")
        return {"error": "Module de corrélation non disponible"}

    if methods is None:
        methods = ["pearson", "spearman", "kendall", "mutual_info"]

    print("\n" + "="*60)
    print("       ANALYSE DES CORRÉLATIONS")
    print("="*60)

    try:
        analyzer = FeatureCorrelationAnalyzer(df, target_col=target_col, task=task)

        results = {
            "target": target_col,
            "task": task,
            "methods_used": methods,
        }

        # Corrélations classiques (Pearson, Spearman, Kendall)
        if any(m in methods for m in ["pearson", "spearman", "kendall"]):
            print("\n[INFO] Calcul des corrélations classiques (Pearson, Spearman, Kendall)...")
            classical = analyzer.compute_classical_correlations()
            results["classical"] = classical.to_dict(orient="records")
            print(f"  ✓ {len(classical)} features analysées")

        # Mutual Information
        if "mutual_info" in methods:
            print("[INFO] Calcul de la Mutual Information...")
            mi = analyzer.compute_mutual_info()
            results["mutual_info"] = mi.to_dict(orient="records")
            print(f"  ✓ Mutual Information calculée")

        # MIC (si minepy installé)
        if "mic" in methods:
            print("[INFO] Calcul du MIC (Maximal Information Coefficient)...")
            mic = analyzer.compute_mic_matrix()
            results["mic"] = mic.to_dict(orient="records")
            if mic["mic"].sum() > 0:
                print(f"  ✓ MIC calculé")
            else:
                print(f"  ⚠ MIC non disponible (minepy non installé)")

        # PhiK (si phik installé)
        if "phik" in methods:
            print("[INFO] Calcul de PhiK...")
            phik = analyzer.compute_phik()
            results["phik"] = phik.to_dict(orient="records")
            if phik["phik"].sum() > 0:
                print(f"  ✓ PhiK calculé")
            else:
                print(f"  ⚠ PhiK non disponible (phik non installé)")

        # Score combiné
        print("[INFO] Calcul du score combiné...")
        combined = analyzer.combined_feature_score(normalize=True)
        results["combined_scores"] = combined.to_dict(orient="records")

        # Top 10 features
        top_10 = combined.head(10)[["feature", "combined_score"]].to_dict(orient="records")
        results["top_10_features"] = top_10

        print("\n" + "-"*60)
        print("TOP 10 FEATURES (par score combiné de corrélation)")
        print("-"*60)
        for i, row in enumerate(top_10, 1):
            print(f"  {i:2d}. {row['feature']:<30} score: {row['combined_score']:.4f}")
        print("-"*60)

        return results

    except Exception as e:
        print(f"[ERROR] Erreur lors du calcul des corrélations: {e}")
        return {"error": str(e)}


def _compute_basic_dataset_stats(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> BasicDatasetStats:
    """
    Calcule des stats globales simples du dataset pour le snapshot LLM.
    """
    n_rows, n_columns = df.shape

    total_missing = int(df.isna().sum().sum())
    missing_ratio = float(total_missing / (n_rows * n_columns)) if n_rows > 0 else 0.0

    n_duplicate_rows = int(df.duplicated().sum())
    duplicate_ratio = float(n_duplicate_rows / n_rows) if n_rows > 0 else 0.0

    # Comptages par type (assez grossiers mais suffisent pour le LLM)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    text_like_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" and c not in datetime_cols
    ]

    # Ici on ne distingue pas finement "catégoriel" vs "texte",
    # tu pourras raffiner si besoin.
    n_numeric_features = len(numeric_cols)
    n_datetime_features = len(datetime_cols)
    n_text_features = len(text_like_cols)
    n_categorical_features = max(0, len(feature_cols) - n_numeric_features - n_datetime_features)

    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    return BasicDatasetStats(
        n_rows=n_rows,
        n_columns=n_columns,
        n_features=len(feature_cols),
        n_numeric_features=n_numeric_features,
        n_categorical_features=n_categorical_features,
        n_text_features=n_text_features,
        n_datetime_features=n_datetime_features,
        total_missing_cells=total_missing,
        missing_cell_ratio=missing_ratio,
        n_duplicate_rows=n_duplicate_rows,
        duplicate_row_ratio=duplicate_ratio,
        memory_mb=memory_mb,
    )


def analyze_dataset_for_fe(
    df: pd.DataFrame,
    target_cols: Union[str, Sequence[str]],
    config: FEAnalysisConfig | None = None,
    print_report: bool = True,
    *,
    # 👇 métadonnées optionnelles pour le LLM
    dataset_name: str = "dataset",
    business_description: Optional[str] = None,
    # 👇 Options pour les corrélations
    with_correlations: bool = False,
    correlation_methods: Optional[List[str]] = None,
    correlation_task: str = "classification",
) -> Dict[str, Any]:
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    else:
        target_cols = list(target_cols)

    for t in target_cols:
        if t not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(
                f"Cible '{t}' absente du DataFrame.\n"
                f"Colonnes disponibles ({len(available_cols)}): {available_cols}"
            )

    if not target_cols:
        raise ValueError("Aucune colonne cible fournie.")

    main_target = target_cols[0]

    config = config or FEAnalysisConfig()
    n_rows = len(df)
    feature_cols = [c for c in df.columns if c not in target_cols]

    # ----------------------------------------------------------------------
    # 1) Analyse cibles (dict classique + dataclass LLM)
    # ----------------------------------------------------------------------
    targets_result = analyze_targets(df, target_cols, config)
    targets_info = targets_result["summary"]               # pour ton rapport humain
    llm_targets: Dict[str, TargetSummaryForLLM] = targets_result["llm"]

    # ----------------------------------------------------------------------
    # 2) Analyse features (dict classique + dataclasses LLM)
    # ----------------------------------------------------------------------
    feat_result = analyze_features(df, feature_cols, config)
    features_info = feat_result["features"]
    warnings = feat_result["warnings"]
    llm_features: Dict[str, FeatureSummaryForLLM] = feat_result["llm_features"]

    # ----------------------------------------------------------------------
    # 3) Leakage (summary + dataclasses LLM, si tu as adapté detect_leakage)
    # ----------------------------------------------------------------------
    leakage_result = detect_leakage(df, feature_cols, target_cols, config)
    suspected_leakage = leakage_result["summary"]          # legacy pour rapport
    llm_leakage = leakage_result.get("llm", [])            # liste de dataclasses ou vide

    # ----------------------------------------------------------------------
    # 3.5) Corrélations (optionnel - activé avec with_correlations=True)
    # ----------------------------------------------------------------------
    correlations_result = None
    if with_correlations:
        correlations_result = _compute_correlations(
            df=df,
            target_col=main_target,
            task=correlation_task,
            methods=correlation_methods,
        )

    # ----------------------------------------------------------------------
    # 4) Rapport "classique" (comme avant)
    # ----------------------------------------------------------------------
    report: Dict[str, Any] = {
        "global": {
            "n_rows": n_rows,
            "n_features": len(feature_cols),
            "n_targets": len(target_cols),
            "feature_cols": feature_cols,
            "target_cols": target_cols,
        },
        "targets": targets_info,
        "features": features_info,
        "suspected_leakage": suspected_leakage,
        "warnings": warnings,
    }

    # Ajouter les corrélations si calculées
    if correlations_result is not None:
        report["correlations"] = correlations_result

    # ----------------------------------------------------------------------
    # 5) Construction du snapshot LLM complet
    # ----------------------------------------------------------------------
    # 5.1 Contexte dataset pour le LLM
    context = DatasetContextForLLM(
        name=dataset_name,
        business_description=business_description,
        metric=getattr(config, "metric", None),
        is_time_dependent=getattr(config, "is_time_dependent", False),
        time_column=getattr(config, "time_column", None),
        primary_keys=list(getattr(config, "primary_keys", [])),
        group_keys=list(getattr(config, "group_keys", [])),
    )

    # 5.2 Stats globales dataset
    basic_stats = _compute_basic_dataset_stats(df, feature_cols)

    # 5.3 Cible principale pour le snapshot (si plusieurs cibles → on prend la première)
    target_llm = llm_targets[main_target]

    # 5.4 Config d'analyse (seuils utilisés pour les flags)
    analysis_cfg: Dict[str, Any] = {
        "max_unique_cat_low": getattr(config, "max_unique_cat_low", None),
        "high_cardinality_threshold": getattr(config, "high_cardinality_threshold", None),
        "id_unique_ratio_threshold": getattr(config, "id_unique_ratio_threshold", None),
        "text_unique_ratio_threshold": getattr(config, "text_unique_ratio_threshold", None),
        "high_missing_threshold": getattr(config, "high_missing_threshold", None),
        "strong_corr_threshold": getattr(config, "strong_corr_threshold", None),
    }

    # 5.5 Notes globales pour le LLM (on y met par ex. les infos de fuite détectée)
    global_notes: List[str] = []
    if len(target_cols) > 1:
        global_notes.append(
            f"Plusieurs cibles détectées ({target_cols}). Le snapshot LLM utilise '{main_target}' comme cible principale."
        )

    for leak in llm_leakage:
        try:
            # leak est une dataclass LeakageSignalForLLM si tu as suivi l'étape précédente
            global_notes.append(
                f"Possible fuite : feature '{leak.feature}' très corrélée à la cible '{leak.target}' "
                f"(corr={leak.correlation:.3f})."
            )
        except AttributeError:
            # si ce n'est pas une dataclass mais un dict
            global_notes.append(
                f"Possible fuite : feature '{leak.get('feature')}' très corrélée à la cible "
                f"'{leak.get('target')}' (corr={leak.get('correlation')})."
            )

    snapshot = FEDatasetSnapshotForLLM(
        context=context,
        basic_stats=basic_stats,
        target=target_llm,
        features=list(llm_features.values()),
        analysis_config=analysis_cfg,
        global_notes=global_notes,
    )

    # On ajoute le snapshot et le payload JSON-friendly dans le report
    report["llm_snapshot"] = snapshot
    report["llm_payload"] = snapshot.to_llm_payload()

    # Ajouter les corrélations au llm_payload si elles ont été calculées
    if correlations_result is not None and "error" not in correlations_result:
        report["llm_payload"]["correlations"] = correlations_result

    # ----------------------------------------------------------------------
    # 6) Impression rapport humain
    # ----------------------------------------------------------------------
    if print_report:
        print_fe_report(report)

    return report
