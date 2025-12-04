# app.py
import json
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import requests


# --------------------------
# Fonctions LLM / Prompt
# --------------------------
def build_prompt(text: str, labels: list[str], add_other: bool, instructions: str = "") -> str:
    allowed = labels.copy()
    if add_other and "Autre" not in allowed:
        allowed.append("Autre")

    base = [
        "Tu es un classifieur strict. Tâche: attribuer EXACTEMENT UNE étiquette parmi la liste autorisée.",
        "Si le texte est ambigu ou hors périmètre, choisis 'Autre' (si disponible).",
        "Réponds UNIQUEMENT avec un JSON valide, sans texte autour.",
        'Format: {"label":"<label>", "confidence":0-1, "justification_bref":"..."}',
        "",
        ("Instructions supplémentaires: " + instructions) if instructions else "",
        "",
        f"ÉTIQUETTES AUTORISÉES: {allowed}",
        "",
        "TEXTE À CLASSER:",
        "<<<",
        text.strip(),
        ">>>",
        "",
        "RAPPEL: pas d'explications hors du JSON. Un seul objet JSON.",
    ]
    return "\n".join([l for l in base if l is not None])


def call_ollama(
    ollama_url: str,
    model: str,
    prompt: str,
    seed: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "options": {"seed": seed, "temperature": temperature, "top_p": top_p},
        "stream": False,
    }
    r = requests.post(ollama_url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_json_strict(resp_text: str) -> dict[str, Any]:
    try:
        return json.loads(resp_text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", resp_text, flags=re.S)
        if not m:
            raise ValueError(f"Réponse non parsable: {resp_text[:200]!r}")
        return json.loads(m.group(0))


def sanitize_output(
    obj: dict[str, Any], allowed: list[str], add_other: bool, threshold: float
) -> dict[str, Any]:
    label = str(obj.get("label", "")).strip()
    conf = float(obj.get("confidence", 0.0))
    just = str(obj.get("justification_bref", "")).strip()

    if label not in allowed:
        if add_other and "Autre" in allowed:
            label = "Autre"
        else:
            label = allowed[0] if allowed else "NA"

    if conf < threshold and add_other and "Autre" in allowed and label != "Autre":
        label = "Autre"

    conf = max(0.0, min(1.0, conf))
    return {"label": label, "confidence": conf, "justification_bref": just}


def classify_once(
    text: str,
    labels: list[str],
    add_other: bool,
    threshold: float,
    ollama_url: str,
    model: str,
    seed: int,
    temperature: float,
    top_p: float,
    timeout: int,
    extra_instructions: str,
) -> dict[str, Any]:
    prompt = build_prompt(text, labels, add_other, extra_instructions)
    raw = call_ollama(ollama_url, model, prompt, seed, temperature, top_p, timeout)
    obj = parse_json_strict(raw.get("response", ""))
    return sanitize_output(obj, labels + (["Autre"] if add_other else []), add_other, threshold)


def classify_vote(
    text: str,
    labels: list[str],
    add_other: bool,
    threshold: float,
    ollama_url: str,
    model: str,
    n: int,
    base_seed: int,
    temperature: float,
    top_p: float,
    timeout: int,
    extra_instructions: str,
) -> dict[str, Any]:
    votes = Counter()
    confs = defaultdict(list)
    details = []
    for i in range(n):
        seed = base_seed + i * 9973
        out = classify_once(
            text,
            labels,
            add_other,
            threshold,
            ollama_url,
            model,
            seed,
            temperature,
            top_p,
            timeout,
            extra_instructions,
        )
        votes[out["label"]] += 1
        confs[out["label"]].append(out["confidence"])
        details.append(out)
    # gagnant
    best_label, best_votes = None, -1
    for lbl, v in votes.items():
        if v > best_votes:
            best_label, best_votes = lbl, v
    tied = [lbl for lbl, v in votes.items() if v == best_votes]
    if len(tied) > 1:
        best_label = max(tied, key=lambda lbl: np.mean(confs[lbl]) if len(confs[lbl]) else 0.0)
    avg_conf = float(np.mean(confs[best_label])) if confs[best_label] else 0.0
    return {
        "label": best_label,
        "confidence": round(avg_conf, 4),
        "votes": dict(votes),
        "details": details,
    }
