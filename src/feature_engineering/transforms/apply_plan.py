from src.feature_engineering.transforms.registry import TRANSFORM_REGISTRY

def apply_llm_feature_plan(df, feature_plan, colonne_vise):
    """
    Applique un plan de feature engineering généré par LLM.

    TODO: Implémenter execute_transformation quand les transformations seront définies.
    """
    df = df.copy()
    logs = []

    for spec in feature_plan:
        # TODO: Implémenter execute_transformation
        log = {"spec": spec, "status": "not_implemented"}
        logs.append(log)

    return df, logs

