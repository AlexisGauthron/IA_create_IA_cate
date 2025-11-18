from __future__ import annotations
from typing import Any, Dict, Union
from copy import deepcopy
import json


def apply_llm_business_annotations(
    snapshot: Dict[str, Any],
    llm_result_raw: Union[str, Dict[str, Any]],
    *,
    min_confidence: float = 0.6,
) -> Dict[str, Any]:
    """
    snapshot : ton JSON compact (dict) du dataset
    llm_result_raw : soit un dict déjà parsé, soit une string JSON comme :

        {"Mode":"Final","context":{...},"features":[...]}

    On renvoie un *nouveau* snapshot avec :
      - context.business_description rempli,
      - context.metric rempli,
      - features[*].feature_description rempli (par "name"),
    si confidence >= min_confidence.

    Comportement spécial :
    - Si 'context' est absent dans la réponse LLM,
      on cherche 'business_description' et 'metric' à la racine
      et on affiche un petit warning.
    """

    # 0) Parser la sortie LLM si c'est une string JSON
    if isinstance(llm_result_raw, str):
        llm_result = json.loads(llm_result_raw)
    else:
        llm_result = llm_result_raw

    out = deepcopy(snapshot)

    # -----------------------------
    # 1) Contexte global
    # -----------------------------
    ctx_out = out.setdefault("context", {})

    ctx_result = llm_result.get("context")

    # ---- business_description & metric selon présence de context ----
    if isinstance(ctx_result, dict):
        # Cas normal : on lit uniquement dans "context"
        bd_src = ctx_result
        metric_src = ctx_result
    else:
        # Fallback : context manquant → on cherche à la racine
        print(
            "[WARN] Réponse LLM sans champ 'context' : "
            "tentative de récupération de 'business_description' et 'metric' à la racine."
        )
        bd_src = llm_result
        metric_src = llm_result

    # business_description
    bd = bd_src.get("business_description")
    if isinstance(bd, dict):
        value = bd.get("value")
        conf = bd.get("confidence", 1.0)
        if value and conf >= min_confidence:
            ctx_out["business_description"] = value
            ctx_out.setdefault("_llm_meta", {})["business_description_confidence"] = conf

    # metric
    metric = metric_src.get("metric")
    if isinstance(metric, dict):
        value = metric.get("value")
        conf = metric.get("confidence", 1.0)
        if value and conf >= min_confidence:
            ctx_out["metric"] = value
            ctx_out.setdefault("_llm_meta", {})["metric_confidence"] = conf

    # -----------------------------
    # 2) Features
    # -----------------------------
    features_out = out.setdefault("features", [])
    # index des features par name dans le snapshot
    feat_by_name = {f.get("name"): f for f in features_out if isinstance(f, dict)}

    for f_res in llm_result.get("features", []) or []:
        name = f_res.get("name")
        if not name:
            continue

        fd = f_res.get("feature_description")
        if not isinstance(fd, dict):
            continue

        value = fd.get("value")
        conf = fd.get("confidence", 1.0)

        if not value or conf < min_confidence:
            continue

        # Si la feature existe déjà dans le snapshot, on la met à jour
        if name in feat_by_name:
            feat = feat_by_name[name]
        else:
            # au cas où le LLM décrit une feature non présente
            feat = {"name": name}
            features_out.append(feat)
            feat_by_name[name] = feat

        feat["feature_description"] = value
        meta = feat.setdefault("_llm_meta", {})
        meta["feature_description_confidence"] = conf

    return out
