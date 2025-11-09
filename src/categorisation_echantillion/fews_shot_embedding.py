# src/categorisation_echantillion/embeddings_proto.py
import os

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
_USE_E5_PREFIX = False
_MODEL_NAME = None

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def load_model(name: str = "intfloat/multilingual-e5-base"):
    """Charge et mémorise le modèle d'embeddings de manière sécurisée."""

    import torch
    torch.set_num_threads(1)

    global _model, _USE_E5_PREFIX, _MODEL_NAME
    try:
        # Si le modèle demandé est déjà chargé, on le retourne directement
        if _MODEL_NAME == name and _model is not None:
            return _model

        _MODEL_NAME = name
        _USE_E5_PREFIX = name.startswith(("intfloat/", "e5", "gte"))

        
        import time
        # Chargement du modèles
        for _ in range(3):
            try:
                print("Chargement Model ...\n ")
                _model = SentenceTransformer(name, device='cpu')
                print("Réussie\n")
                break
            except RuntimeError as e:
                print("Retrying model load due to RuntimeError:", e)
                time.sleep(1)


        # On vide le cache de _embed_one_cached si on change de modèle
        _embed_one_cached.cache_clear()
        return _model

    except RuntimeError as e:
        print(f"Erreur lors du chargement du modèle (RuntimeError) : {e}")
    except Exception as e:
        print(f"Erreur inattendue lors du chargement du modèle : {e}")

    # Si échec, on retourne None et on évite le crash
    _model = None
    return _model



def _prep_text(txt: str, is_query: bool) -> str:
    if _USE_E5_PREFIX:
        return f"{'query' if is_query else 'passage'}: {txt}"
    return txt


@lru_cache(maxsize=8192)
def _embed_one_cached(txt: str, is_query: bool) -> np.ndarray:
    if _model is None:
        load_model()  # par défaut
    v = _model.encode([_prep_text(txt, is_query)], normalize_embeddings=True)[0]
    return v.astype(np.float32)


def embed_texts(texts: List[str], is_query: bool) -> np.ndarray:
    if _model is None:
        load_model()
    prepped = [_prep_text(t, is_query) for t in texts]
    vs = _model.encode(prepped, normalize_embeddings=True)
    return vs.astype(np.float32)


def build_prototypes(
    shots: Dict[str, List[str]],
    label_defs: Optional[Dict[str, str]] = None,
    alpha: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Prototype = moyenne des embeddings d'exemples (+ mélange avec définition pondérée par alpha)."""
    protos = {}
    for lbl, examples in shots.items():
        if not examples:
            continue
        ex_vecs = embed_texts(examples, is_query=False)
        proto = ex_vecs.mean(axis=0)
        if label_defs and label_defs.get(lbl):
            d_vec = _embed_one_cached(label_defs[lbl], is_query=False)
            proto = (1 - alpha) * proto + alpha * d_vec
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        protos[lbl] = proto.astype(np.float32)
    return protos


# Pour chaque prototype, on garde les labels dont la similarité dépasse le seuil.
# Permet d’attribuer plusieurs labels à un texte.
# Si aucun label ne passe le seuil → "Autre".




def classify_one(
    text: str,
    protos: Dict[str, np.ndarray],
    threshold: float = 0.35,
    margin: float = 0.05,
    allow_other: bool = True,
) -> Dict:
    """Mono-label : top-1 par cosinus, rejet 'Autre' si top<seuil ou marge top1-top2 faible."""
    v = _embed_one_cached(text, is_query=True)
    if not protos:
        return {"label": "Autre", "confidence": 0.0, "sims": {}}
    labels = list(protos.keys())
    mats = np.stack([protos[l] for l in labels])
    sims = (mats @ v)
    order = np.argsort(-sims)
    top, second = order[0], (order[1] if len(order) > 1 else order[0])
    top_lbl, top_sim = labels[top], float(sims[top])
    second_sim = float(sims[second])
    if allow_other and (top_sim < threshold or (top_sim - second_sim) < margin):
        return {"label": "Autre", "confidence": max(0.0, min(1.0, top_sim)), "sims": dict(zip(labels, sims.tolist()))}
    return {"label": top_lbl, "confidence": max(0.0, min(1.0, top_sim)), "sims": dict(zip(labels, sims.tolist()))}


# Calcule l’embedding du texte.
# Compare avec tous les prototypes via similarité cosinus (dot product, car vecteurs normalisés).
# Règles de décision :
# top_sim < threshold → label = "Autre".
# (top_sim - second_sim) < margin → label = "Autre" (texte ambigu).
# Retourne : {"label": ..., "confidence": score_top, "sims": {label: score}}.



def classify_one_multi(
    text: str,
    protos: Dict[str, np.ndarray],
    per_label_threshold: float = 0.4
) -> Dict:
    """Multi-label : conserve toutes les classes dont la similarité dépasse un seuil."""
    v = _embed_one_cached(text, is_query=True)
    sims = {lbl: float(vec @ v) for lbl, vec in protos.items()}
    kept = [lbl for lbl, s in sims.items() if s >= per_label_threshold]
    kept.sort(key=lambda k: sims[k], reverse=True)
    return {"labels": kept if kept else ["Autre"], "sims": sims}


def calibrate_threshold(
    shots: Dict[str, List[str]],
    label_defs: Optional[Dict[str, str]] = None,
    alpha: float = 0.3
) -> Tuple[float, float]:
    """Calibre (threshold, margin) par leave-one-out sur vos exemples."""
    pos_sims, margins = [], []
    for lbl, examples in shots.items():
        if len(examples) < 2:
            continue
        for i, ex in enumerate(examples):
            others = [t for j, t in enumerate(examples) if j != i]
            tmp_shots = {k: (v if k != lbl else others) for k, v in shots.items()}
            protos = build_prototypes(tmp_shots, label_defs, alpha)
            # collecte des stats si bien reconnu
            vq = _embed_one_cached(ex, is_query=True)
            if not protos:
                continue
            labels = list(protos.keys())
            mats = np.stack([protos[l] for l in labels])
            sims = mats @ vq
            order = np.argsort(-sims)
            if labels[order[0]] == lbl:
                top_sim = float(sims[order[0]])
                second_sim = float(sims[order[1]]) if len(order) > 1 else float(sims[order[0]])
                pos_sims.append(top_sim)
                margins.append(top_sim - second_sim)
    thr = float(np.percentile(pos_sims, 10)) if pos_sims else 0.35
    mar = float(np.percentile(margins, 10)) if margins else 0.05
    thr = float(np.clip(thr, 0.2, 0.6))
    mar = float(np.clip(mar, 0.02, 0.15))
    return thr, mar


if __name__ == "__main__":
    # Petit banc d'essai autonome pour les fonctions du module.
    # Exécuter:  python -m src.categorisation_echantillion.embeddings_proto
    import os
    import textwrap

    # Option: changer de modèle via la variable d'env EMB_MODEL_NAME
    model_name = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-base")
    print(f"[Init] Chargement du modèle: {model_name}")
    load_model(model_name)

    # --------- Jeu d'exemples (few-shots) par classe ----------
    shots = {
        "Support Technique": [
            "L'application plante à l'ouverture après la mise à jour.",
            "Je n'arrive plus à me connecter, le mot de passe est refusé.",
            "Erreur 500 sur la page d'accueil du site.",
            "Le serveur est indisponible depuis ce matin.",
        ],
        "Facturation": [
            "Pouvez-vous renvoyer la facture de septembre avec la TVA corrigée ?",
            "Merci d'émettre un avoir pour la facture #4589.",
            "Quand le paiement par virement sera-t-il enregistré ?",
            "Je souhaite un devis pour renouveler mon abonnement.",
        ],
        "Ressources humaines": [
            "Je pose deux jours de congés la semaine prochaine.",
            "Mon bulletin de paie comporte une erreur de prime.",
            "Comment déclarer un arrêt maladie de trois jours ?",
            "J'ai besoin d'un avenant à mon contrat de travail.",
        ],
        "Logistique": [
            "Mon colis est bloqué au dépôt depuis trois jours.",
            "Le transporteur indique une adresse de livraison incomplète.",
            "Quel est le délai d'expédition pour cette commande ?",
            "Le suivi indique que le paquet est perdu.",
        ],
        "Commercial": [
            "Auriez-vous une remise pour 50 licences ?",
            "Quand est prévue la prochaine démo produit ?",
            "Pouvez-vous m'envoyer une proposition commerciale détaillée ?",
            "Je souhaite discuter du prix et des options.",
        ],
    }

    # --------- Définitions textuelles (améliorent les prototypes via alpha) ----------
    label_defs = {
        "Support Technique": "Problèmes techniques, bugs, erreurs, connexion, mises à jour, mot de passe, serveur, application.",
        "Facturation": "Factures, paiements, TVA, avoirs, remboursements, devis, comptabilité.",
        "Ressources humaines": "Congés, arrêts maladie, contrat de travail, bulletin de paie, recrutement.",
        "Logistique": "Livraison, colis, transporteur, entrepôt, expédition, délai, adresse de livraison.",
        "Commercial": "Prospection, devis, prix, remise, démonstration produit, négociation, vente.",
    }

    print("[Build] Construction des prototypes…")
    protos = build_prototypes(shots, label_defs=label_defs, alpha=0.30)

    print("[Calib] Calibration des hyperparamètres de rejet (threshold, margin)…")
    thr, mar = calibrate_threshold(shots, label_defs=label_defs, alpha=0.30)
    print(f"[Calib] threshold={thr:.3f} | margin={mar:.3f}")

    # --------- Jeu de tests mono-label ----------
    tests = [
        ("Je n'arrive plus à me connecter après la mise à jour.", "Support Technique"),
        ("Pouvez-vous me renvoyer la facture de septembre avec la TVA corrigée ?", "Facturation"),
        ("Mon colis est bloqué au dépôt depuis trois jours.", "Logistique"),
        ("Je souhaite poser deux jours de congés la semaine prochaine.", "Ressources humaines"),
        ("Auriez-vous une remise pour 50 licences ?", "Commercial"),
        ("Le serveur affiche une erreur 500 sur la page d'accueil.", "Support Technique"),
        ("Merci d'annuler la facture #4589 et d'émettre un avoir.", "Facturation"),
        ("Quand est prévue la prochaine démo produit ?", "Commercial"),
        ("Mon bulletin de paie comporte une erreur de prime.", "Ressources humaines"),
        ("Le transporteur m'indique une adresse incomplète pour la livraison.", "Logistique"),
    ]

    print("\n===== Test mono-label (classify_one) =====")
    ok = 0
    for text, expected in tests:
        res = classify_one(text, protos, threshold=thr, margin=mar, allow_other=True)
        pred, conf = res["label"], res["confidence"]
        mark = "OK" if pred == expected else "!!"
        top3 = sorted(res["sims"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        sims_str = ", ".join([f"{k}:{v:.2f}" for k, v in top3])
        print(f"[{mark}] y={expected:<18} → ŷ={pred:<18} (conf={conf:.2f}) | top3= {sims_str}")
        ok += int(pred == expected)
    print(f"[Score] Exact-match accuracy: {ok}/{len(tests)} = {ok/len(tests):.1%}")

    # --------- Démo multi-label ----------
    multi_tests = [
        "Le colis est perdu et j'ai besoin d'un avoir.",
        "Après la démo, pouvez-vous m'envoyer le devis ?",
        "Impossible de me connecter pour récupérer ma facture.",
        "Je dois modifier l'adresse de livraison et connaître le prix.",
    ]
    ml_thr = max(0.40, thr)  # seuil raisonnable pour la multi-étiquette
    print("\n===== Test multi-label (classify_one_multi) =====")
    for text in multi_tests:
        res = classify_one_multi(text, protos, per_label_threshold=ml_thr)
        # Affiche les 3 meilleurs scores pour info
        order = sorted(protos.keys(), key=lambda k: res["sims"][k], reverse=True)
        top3 = [(k, res["sims"][k]) for k in order[:3]]
        print("- " + textwrap.fill(text, width=88))
        print(f"  → labels={res['labels']} | top3=" + ", ".join(f"{k}:{v:.2f}" for k, v in top3))

    print("\n[Fin] Banc d'essai terminé.")
