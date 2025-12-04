from __future__ import annotations

# src/feature_analysis/printing.py
from typing import Any


def print_fe_report(report: dict[str, Any]) -> None:
    """Affichage texte lisible du rapport de feature engineering."""

    g = report["global"]
    print("=" * 80)
    print("RAPPORT D'ANALYSE POUR FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Lignes : {g['n_rows']}")
    print(f"Features : {g['n_features']} | Cibles : {g['n_targets']}")
    print(f"Cibles : {g['target_cols']}")
    print()

    print("=== ANALYSE DES CIBLES ===")
    for t, info in report["targets"].items():
        print(f"\n- Cible : {t}")
        print(f"  Type déduit : {info['target_type']} | Problème : {info['problem_hint']}")
        print(f"  dtype : {info['dtype']} | n_unique : {info['n_unique']}")
        print(f"  Taux de NaN : {info['missing_rate']:.1%}")
        for note in info["notes"]:
            print("   ⚠️ ", note)
        if info["value_counts"] is not None:
            print("  Top modalités / classes :")
            print(info["value_counts"])

    print("\n=== ANALYSE DES FEATURES ===")
    for col, info in report["features"].items():
        print(f"\n- Feature : {col}")
        print(f"  Rôle : {info['role']} | dtype : {info['dtype']}")
        print(f"  n_unique : {info['n_unique']} (ratio {info['unique_ratio']:.1%})")
        print(f"  Taux de NaN : {info['missing_rate']:.1%}")
        flags = []
        if info["is_constant"]:
            flags.append("CONST")
        if info["is_id_like"]:
            flags.append("ID_LIKE")
        if info["high_cardinality"]:
            flags.append("HIGH_CARD")
        if flags:
            print("  Flags :", ", ".join(flags))
        if info["notes"]:
            print("  Notes :")
            for note in info["notes"]:
                print("   -", note)
        if info["recommendations"]:
            print("  Recommandations FE :")
            for rec in info["recommendations"]:
                print("   →", rec)

    if report["suspected_leakage"]:
        print("\n=== CORRÉLATIONS FORTES (POTENTIEL LEAKAGE) ===")
        for item in report["suspected_leakage"]:
            print(
                f"- Feature '{item['feature']}' vs cible '{item['target']}' : "
                f"corr={item['correlation']:.3f} → {item['note']}"
            )

    if report["warnings"]:
        print("\n=== WARNINGS GLOBAUX ===")
        for w in report["warnings"]:
            print(" -", w)

    print("\nFin du rapport.")
    print("=" * 80)
    print("\n\n")
