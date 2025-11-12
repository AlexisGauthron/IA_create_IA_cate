from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

def evaluate_fewshot_models(
    texts: List[str],
    y_true: List[str],
    shots: Dict[str, List[str]],
    *,
    # --- Embeddings proto ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    proto_thr_other: float = 0.0,    # 0.0 => jamais "Autre"; ex: 0.35 pour rejeter bas
    # --- Zero-shot NLI ---
    zshot_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    zshot_multi_label: bool = False,
    zshot_thr_other: float = 0.0,    # seuil d'abstention sur le score top-1
    # --- Ollama ICL ---
    ollama_model: str = "mistral:latest",
    k_per_label: int = 4,
    temperature: float = 0.0,
    seed: Optional[int] = 42,
    # --- Divers ---
    lowercase_match: bool = True,    # robustesse matcher label
    return_predictions: bool = True
) -> Dict[str, Any]:
    """
    Compare plusieurs approches few-shot sur un jeu (texts, y_true) et retourne:
      {
        "summary": pd.DataFrame(scores par modèle),
        "preds":   pd.DataFrame({y_true, <model1>, <model2>, ...})   (si return_predictions=True)
      }

    Approches:
      1) Prototypes d'embeddings (cosine argmax, seuil optionnel 'Autre')
      2) Zero-shot NLI (Transformers pipeline)
      3) Few-shot in-context via Ollama (prompt structuré)
    """
    import warnings
    warnings.filterwarnings("ignore")

    labels = list(shots.keys())
    if lowercase_match:
        # map pour normaliser les libellés produits par les modèles
        lb_map = {l.lower(): l for l in labels}
    else:
        lb_map = {l: l for l in labels}

    # -----------------------------
    #  Helpers: métriques
    # -----------------------------
    from sklearn.metrics import accuracy_score, f1_score

    def _scores(y, yhat, model_name: str) -> Dict[str, float]:
        # Normalise taille & NaN
        y = list(y)
        yhat = list(yhat)
        acc = accuracy_score(y, yhat)
        f1m = f1_score(y, yhat, average="macro", zero_division=0)
        f1w = f1_score(y, yhat, average="weighted", zero_division=0)
        return {"model": model_name, "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w}

    # -----------------------------
    #  1) Embeddings prototypes
    # -----------------------------
    proto_preds = None
    try:
        from sentence_transformers import SentenceTransformer

        emb = SentenceTransformer(embed_model)
        # prototypes
        proto_vecs = {}
        for lbl, exs in shots.items():
            if not exs: 
                continue
            E = emb.encode(exs[:max(1, k_per_label)], normalize_embeddings=True, show_progress_bar=False)
            proto_vecs[lbl] = E.mean(axis=0)
        # normalisation
        for lbl, v in proto_vecs.items():
            v = v / (np.linalg.norm(v) + 1e-9)
            proto_vecs[lbl] = v

        # matrice prototypes
        p_labels = list(proto_vecs.keys())
        P = np.stack([proto_vecs[l] for l in p_labels]) if p_labels else None

        proto_preds = []
        for t in texts:
            v = emb.encode([t], normalize_embeddings=True, show_progress_bar=False)[0]
            if P is None:
                proto_preds.append("Autre")
                continue
            sims = P @ v
            j = int(np.argmax(sims))
            best_sim = float(sims[j])
            pred = p_labels[j] if (best_sim >= proto_thr_other) else "Autre"
            proto_preds.append(pred)

    except Exception as e:
        proto_preds = None  # package manquant ou autre

    # -----------------------------
    #  2) Zero-shot NLI
    # -----------------------------
    zshot_preds = None
    try:
        from transformers import pipeline
        zpipe = pipeline("zero-shot-classification", model=zshot_model)
        zshot_preds = []
        cand = labels  # autorisés
        for t in texts:
            out = zpipe(t, candidate_labels=cand, multi_label=zshot_multi_label)
            # pipeline peut renvoyer dict ou list; on standardise
            if isinstance(out, dict):
                labs = out["labels"]
                scores = out["scores"]
                top = labs[0]
                top_score = scores[0]
            else:
                labs = out[0]["labels"]
                scores = out[0]["scores"]
                top = labs[0]
                top_score = scores[0]

            # normalisation libellé
            key = top.lower() if lowercase_match else top
            pred = lb_map.get(key, top)
            if zshot_thr_other > 0.0 and float(top_score) < zshot_thr_other:
                pred = "Autre"
            zshot_preds.append(pred)
    except Exception:
        zshot_preds = None

    # -----------------------------
    #  3) Few-shot ICL via Ollama
    # -----------------------------
    ollama_preds = None
    try:
        import ollama

        def _mk_prompt(label_examples: Dict[str, List[str]]) -> str:
            # Construit un contexte few-shot compact
            lines = []
            for l in labels:
                exs = (label_examples.get(l) or [])[:k_per_label]
                if not exs:
                    continue
                lines.append(f"Label: {l}\n" + "\n".join([f"- {e}" for e in exs]))
            guide = (
                "Tu es un classifieur. À partir des exemples ci-dessus, "
                "attribue **exactement** l'un des labels suivants au texte cible. "
                f"Labels autorisés: {', '.join(labels)}. "
                "Réponds par **un seul label** (et rien d'autre)."
            )
            return "\n\n".join(lines) + "\n\n" + guide

        sys_msg = "Tu es un assistant concis. Réponds uniquement par le nom du label demandé."
        context = _mk_prompt(shots)

        ollama_preds = []
        for t in texts:
            user_msg = f"Texte cible:\n{t}\n\nLabel:"
            resp = ollama.chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": context},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": float(temperature), **({"seed": int(seed)} if seed is not None else {})},
            )
            out = resp["message"]["content"].strip()
            out_norm = out.lower().strip() if lowercase_match else out.strip()
            # match au plus proche (exacte ou début de chaîne)
            pred = None
            for L in labels:
                key = L.lower() if lowercase_match else L
                if out_norm == key or out_norm.startswith(key):
                    pred = L
                    break
            if pred is None:
                pred = "Autre"
            ollama_preds.append(pred)
    except Exception:
        ollama_preds = None

    # -----------------------------
    #  Résumé des scores
    # -----------------------------
    rows = []
    pred_cols = {}
    if proto_preds is not None:
        rows.append(_scores(y_true, proto_preds, f"proto@{embed_model.split('/')[-1]}(thr={proto_thr_other})"))
        pred_cols["proto"] = proto_preds
    if zshot_preds is not None:
        rows.append(_scores(y_true, zshot_preds, f"zshot@{zshot_model.split('/')[-1]}(thr={zshot_thr_other})"))
        pred_cols["zshot"] = zshot_preds
    if ollama_preds is not None:
        rows.append(_scores(y_true, ollama_preds, f"ollama@{ollama_model}"))
        pred_cols["ollama"] = ollama_preds

    if not rows:
        raise RuntimeError("Aucun backend disponible (installe sentence-transformers, transformers, et/ou lance ollama).")

    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    out = {"summary": summary}
    if return_predictions:
        preds_df = pd.DataFrame({"text": texts, "y_true": y_true, **pred_cols})
        out["preds"] = preds_df
    return out


shots = {
    "Support Technique": [
        "Impossible de se connecter au serveur",
        "Erreur 500 sur l'app web",
        "Bug de synchronisation des données",
        "Mot de passe rejeté depuis la MAJ"
    ],
    "Facturation": [
        "Montant de la facture incorrect",
        "Demande d'avoir et de remboursement",
        "TVA non appliquée",
        "Problème de paiement carte"
    ],
}

texts = [
    "Erreur 502 sur le proxy depuis hier",
    "Pouvez-vous renvoyer la facture d'octobre ?",
    "Le colis n'est pas arrivé",
]
y_true = ["Support Technique", "Facturation", "Autre"]  # exemple

res = evaluate_fewshot_models(
    texts, y_true, shots,
    proto_thr_other=0.35,           # rejeter si cosinus top-1 < 0.35
    zshot_thr_other=0.50,           # rejeter si score top-1 < 0.50
    k_per_label=3,
    ollama_model="mistral:latest",
)

print(res["summary"])     # tableau comparatif des scores
print(res["preds"].head())
