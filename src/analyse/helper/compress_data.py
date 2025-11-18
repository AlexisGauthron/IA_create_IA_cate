from __future__ import annotations

from typing import Any, Dict, List
from copy import deepcopy
import math

import src.analyse.helper.suppression_vnul as suppression_vnul


def _round_floats(obj: Any, ndigits: int = 4) -> Any:
    """Arrondit récursivement tous les floats pour réduire les tokens."""
    if isinstance(obj, float):
        # gère les NaN / inf
        if math.isnan(obj) or math.isinf(obj):
            return obj
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def compact_llm_snapshot_payload(
    payload: Dict[str, Any],
    *,
    max_example_values: int = 3,
    max_top_values: int = 3,
    float_ndigits: int = 4,
    feature_engineering: bool = True,
) -> Dict[str, Any]:
    """
    Prend un snapshot LLM (dict) et le compresse pour réduire
    le nombre de tokens sans perdre l'info importante pour le FE.
    """
    data = deepcopy(payload)

    # -----------------------------
    # 0) context : retirer champs pas utiles pour le FE
    # -----------------------------
    ctx = data.get("context") or {}
    if ctx:
        # name = juste label humain, pas nécessaire pour le LLM FE
        ctx.pop("name", None)

        # primary_keys / group_keys : on les supprime si vides
        if ctx.get("primary_keys") == []:
            ctx.pop("primary_keys", None)
        if ctx.get("group_keys") == []:
            ctx.pop("group_keys", None)

        # is_time_dependent = utile surtout si True, sinon on peut l'omettre
        if not ctx.get("is_time_dependent", False):
            ctx.pop("is_time_dependent", None)

        data["context"] = ctx

    # -----------------------------
    # 1) basic_stats : on garde l’essentiel
    # -----------------------------
    bs = data.get("basic_stats", {})
    # n_features & co sont dérivables depuis features → on les enlève
    keep_bs = [
        "n_rows",
        "missing_cell_ratio",
    ]
    data["basic_stats"] = {k: bs[k] for k in keep_bs if k in bs}

    # -----------------------------
    # 2) target : on tronque/agresse un peu
    # -----------------------------
    target = data.get("target", {})
    if target:
        # On supprime les champs jugés redondants ou peu utiles
        target.pop("pandas_dtype", None)
        target.pop("inferred_target_type", None)
        target.pop("n_rows", None)
        target.pop("n_unique", None)
        target.pop("most_frequent_classes", None)
        target.pop("imbalance_ratio", None)
        target.pop("notes", None)
        target.pop("is_imbalanced", None)

        # missing_rate = utile si > 0, sinon on le supprime
        if target.get("missing_rate", 0.0) == 0.0:
            target.pop("missing_rate", None)

        # On garde seulement class_counts et on supprime class_proportions
        target.pop("class_proportions", None)

        data["target"] = target

    # -----------------------------
    # 3) features : grosse source de tokens → on compacte
    # -----------------------------
    new_features: List[Dict[str, Any]] = []
    for feat in data.get("features", []):
        f = dict(feat)  # shallow copy

        inferred_type = f.get("inferred_type")

        # 3.0 Retirer quelques champs génériques
        f.pop("pandas_dtype", None)
        f.pop("n_rows", None)
        f.pop("role", None)          # redondant avec inferred_type
        f.pop("unique_ratio", None)  # n_unique + n_rows suffisent

        # notes = redondant avec flags + fe_hints → on les enlève
        f.pop("notes", None)

        # Si on n'est PAS en mode feature engineering, on supprime fe_hints
        if not feature_engineering:
            f.pop("fe_hints", None)

        # missing_rate : on le conserve seulement s'il est > 0
        if f.get("missing_rate", 0.0) == 0.0:
            f.pop("missing_rate", None)

        if not feature_engineering:
            # 3.1 Tronquer / supprimer example_values
            ex_vals = f.get("example_values")
            if isinstance(ex_vals, list):
                # Pour les features numériques, exemples peu utiles (on a déjà les stats)
                if inferred_type == "numeric":
                    f.pop("example_values", None)
                else:
                    flags = f.get("flags") or []
                    if "ID_LIKE" in flags:
                        # pour les ID-like, les exemples n'apportent rien
                        f.pop("example_values", None)
                    else:
                        # max N exemples uniques
                        seen = set()
                        uniq = []
                        for v in ex_vals:
                            if v not in seen:
                                uniq.append(v)
                                seen.add(v)
                            if len(uniq) >= max_example_values:
                                break
                        if uniq:
                            f["example_values"] = uniq
                        else:
                            f.pop("example_values", None)

        # 3.2 Compactage de numeric_stats
        num_stats = f.get("numeric_stats")
        if isinstance(num_stats, dict):
            # on garde les clés les plus utiles pour le FE
            keep_num = ["mean", "std", "min", "max", "skewness"]
            f["numeric_stats"] = {
                k: num_stats[k] for k in keep_num if k in num_stats
            }
        else:
            # si pas de stats, on supprime carrément la clé
            f.pop("numeric_stats", None)

        # 3.3 Compactage de categorical_stats
        cat_stats = f.get("categorical_stats")
        if isinstance(cat_stats, dict):
            new_cat: Dict[str, Any] = {}

            n_rare = cat_stats.get("n_rare_levels")
            if n_rare not in (None, 0):
                new_cat["n_rare_levels"] = n_rare

            # Tronquer top_values
            tv = cat_stats.get("top_values")
            if isinstance(tv, list) and len(tv) > 0:
                new_cat["top_values"] = tv[:max_top_values]

            if new_cat:
                f["categorical_stats"] = new_cat
            else:
                f.pop("categorical_stats", None)
        else:
            f.pop("categorical_stats", None)

        # 3.4 text_stats : on garde tout si présent (mais supprimé si None)
        if not isinstance(f.get("text_stats"), dict):
            f.pop("text_stats", None)

        # 3.5 Retirer certains champs redondants
        for redundant_key in ("n_non_null", "n_missing"):
            f.pop(redundant_key, None)

        new_features.append(f)

    data["features"] = new_features

    # -----------------------------
    # 4) analysis_config / global_notes
    # -----------------------------
    # analysis_config : on le supprime pour économiser des tokens (flags suffisent)
    data.pop("analysis_config", None)

    # global_notes : souvent vide ; si c'est encore le cas, on supprime.
    if not data.get("global_notes"):
        data.pop("global_notes", None)

    # -----------------------------
    # 5) Arrondir tous les floats + supprimer les nulls
    # -----------------------------
    data = _round_floats(data, ndigits=float_ndigits)
    data = suppression_vnul.remove_nulls(data)

    return data
