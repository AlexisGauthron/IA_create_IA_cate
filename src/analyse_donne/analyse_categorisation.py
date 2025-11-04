from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class ClarificationY:
    task_type: str                     # "binary", "multiclass", "multilabel", "non_discrete"
    label_kind: str                    # "int", "float", "str/bool", "mixed"
    n_classes: Optional[int]
    classes: Optional[List[Any]]
    class_distribution: Optional[Dict[Any, float]]  # % par classe
    missing_rate: float
    imbalance_ratio: Optional[float]   # max(p)/min(p)
    suggestions_metrics: List[str]
    notes: List[str]


# Recherche si la séries est discrètes 
def _is_discrete_series(s: pd.Series, max_unique_for_discrete: int = 100) -> bool:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    # floats: on accepte si peu de modalités distinctes
    if pd.api.types.is_float_dtype(s):
        nunique = s.dropna().nunique()
        return nunique <= max_unique_for_discrete
    # objets/strings
    if pd.api.types.is_object_dtype(s):
        return True
    return False


def _infer_label_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "str/bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return "str/bool"
    return "mixed"

# Test format multilabel
def is_binary_col(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        # pour les entiers, on s’assure qu’il n’y a que 0/1 (en ignorant les NaN)
        vals = set(s.dropna().unique().tolist())
        return vals.issubset({0, 1})
    # Autoriser float/obj encodés en 0.0/1.0
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_object_dtype(s):
        vals = set(pd.Series(s).dropna().unique().tolist())
        return vals.issubset({0, 1, 0.0, 1.0})
    return False


def clarifier_objectif_cible(
    df: pd.DataFrame,
    y: Union[str, pd.Series, pd.DataFrame],
    categories_possibles: Optional[List[Any]] = None,
) -> ClarificationY:
    """
    Clarifie l’objectif et la cible (y) :
      - détecte le type de tâche (binaire, multi-classe, multi-label)
      - contrôle la cohérence des étiquettes avec categories_possibles
      - mesure le déséquilibre
      - suggère les bonnes métriques
    """

    notes = []

    # --- Récupère la cible sous forme exploitable ---
    if isinstance(y, str):
        if y not in df.columns:
            raise ValueError(f"[INFO] La colonne cible '{y}' est absente du DataFrame.\n")
        target = df[y]
    else:
        target = y

    # Test si la colonne à prédire est une colonne multilabel 
    is_multilabel = isinstance(target, pd.DataFrame) and all(is_binary_col(target[c]) for c in target.columns)


    if is_multilabel:
        Y = target.fillna(0).astype(int)
        # distribution moyenne par label
        label_prevalence = (Y.mean().to_dict())
        # métriques suggérées
        suggestions = ["F1-micro", "F1-macro", "SubsetAccuracy", "HammingLoss", "AveragePrecision"]
        return ClarificationY(
            task_type="multilabel",
            label_kind="int/bool (multi-colonnes)",
            n_classes=Y.shape[1],
            classes=list(Y.columns),
            class_distribution={k: float(v) for k, v in label_prevalence.items()},
            missing_rate=float(target.isna().mean().mean()),
            imbalance_ratio=float(max(label_prevalence.values()) / max(1e-12, min(v for v in label_prevalence.values() if v > 0))) if any(v > 0 for v in label_prevalence.values()) else None,
            suggestions_metrics=suggestions,
            notes=notes + ["Détection multi-label : chaque colonne représente un label binaire indépendant.",
                           "Pensez au stratified split multilabel (iterative stratification)."]
        )


    # --- Le reste : y est supposé mono-colonne ---
    if isinstance(target, pd.DataFrame):
        # Plusieurs colonnes non-binaires => impossible d’inférer automatiquement
        raise ValueError("Plusieurs colonnes cibles détectées mais non binaires. Précisez le cadre (multilabel, multioutput…).")


    # Cherche s'il y a des valeurs manquantes dans la catégorie à prédire
    s = target
    rate = s.isna().mean()
    missing_rate = 0.0 if pd.isna(rate) else float(rate)
    if missing_rate > 0:
        notes.append(f"Taux de valeurs manquantes dans y : {missing_rate:.2%} (à imputer ou à exclure).")


    # Contrôle des catégories possibles si préciser
    if categories_possibles is not None:
        inconnues = set(s.dropna().unique()) - set(categories_possibles)
        if inconnues:
            notes.append(f"Valeurs de y hors 'categories_possibles' : {sorted(map(str, inconnues))}.")

    # Discret ou non ?
    if not _is_discrete_series(s):
        # Pas une tâche de classification discrète -> probablement régression
        return ClarificationY(
            task_type="non_discrete",
            label_kind=_infer_label_kind(s),
            n_classes=None,
            classes=None,
            class_distribution=None,
            missing_rate=missing_rate,
            imbalance_ratio=None,
            suggestions_metrics=["(régression) MAE", "RMSE", "R²", "MedAE"],
            notes=notes + ["La cible semble continue : ce n’est pas une classification.",
                           "Si vous voulez forcer une classification, définissez des bins (discrétisation)."]
        )


    # Série discrète : binaire vs multi-classe
    values = s.dropna()
    classes = sorted(values.unique(), key=lambda x: str(x))
    n_classes = len(classes)

    if n_classes == 1:
        notes.append("Une seule classe observée : besoin de données supplémentaires ou d’un autre échantillonnage.")
    if n_classes == 2:
        task_type = "binary"
    else:
        task_type = "multiclass"

    # Distribution & déséquilibre
    counts = values.value_counts(normalize=True).to_dict()
    imbalance_ratio = float(max(counts.values()) / max(1e-12, min(counts.values()))) if n_classes >= 2 else None

    # Choix des métriques selon le cadre et l’équilibre
    suggestions: List[str] = []
    if task_type == "binary":
        if imbalance_ratio and imbalance_ratio >= 3:  # seuil simple
            suggestions = ["F1", "ROC-AUC", "PR-AUC", "BalancedAccuracy"]
            notes.append("Déséquilibre marqué : privilégier PR-AUC, F1 et BalancedAccuracy plutôt qu’Accuracy.")
        else:
            suggestions = ["Accuracy", "F1", "ROC-AUC"]
    elif task_type == "multiclass":
        if imbalance_ratio and imbalance_ratio >= 3:
            suggestions = ["F1-macro", "BalancedAccuracy", "MCC", "Top-k Accuracy"]
            notes.append("Déséquilibre multi-classe : utiliser F1-macro / BalancedAccuracy.")
        else:
            suggestions = ["Accuracy", "F1-macro", "LogLoss", "MCC"]

    return ClarificationY(
        task_type=task_type,
        label_kind=_infer_label_kind(s),
        n_classes=n_classes,
        classes=[c for c in classes],
        class_distribution={k: float(v) for k, v in counts.items()},
        missing_rate=missing_rate,
        imbalance_ratio=imbalance_ratio,
        suggestions_metrics=suggestions,
        notes=notes + ["Utilisez un split stratifié (StratifiedKFold) pour conserver les proportions de classes.",
                       "Vérifiez le coût des erreurs (précision vs rappel) pour choisir la métrique clé."]
    )



# --- Exemple d’utilisation ---
if __name__ == "__main__":
    df = pd.DataFrame({
        "text": ["a", "b", "c", "d", "e", "f"],
        "y":    [0, 0, 0, 1, 1, 1]
    })
    clarif = clarifier_objectif_cible(df, "y", categories_possibles=[0,1])
    print("[INFO] Analyse donne :\n",asdict(clarif))
