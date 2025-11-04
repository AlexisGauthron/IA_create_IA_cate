def afficher_clarif(c):
    print("— Clarification de la cible —")
    print(f"• Type de tâche        : {c.task_type}")
    print(f"• Type d’étiquette     : {c.label_kind}")
    if c.n_classes is not None:
        print(f"• Nombre de classes    : {c.n_classes}")
        if c.classes:
            print(f"• Classes              : {list(c.classes)}")
    print(f"• Taux de manquants    : {c.missing_rate:.2%}")
    if c.imbalance_ratio is not None:
        print(f"• Ratio de déséquilibre: {c.imbalance_ratio:.2f}×")
    if c.class_distribution:
        print("• Distribution des classes :")
        for k, v in c.class_distribution.items():
            print(f"   - {k}: {v:.2%}")
    if c.suggestions_metrics:
        print("• Métriques suggérées  : " + ", ".join(c.suggestions_metrics))
    if c.notes:
        print("• Notes :")
        for n in c.notes:
            print(f"   • {n}")

