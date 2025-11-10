import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



from typing import Dict, List, Tuple, Optional
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import src.categorisation_echantillion.f_embedding as f_emb
import src.categorisation_echantillion.calibrate_embedding as cal_emb
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
    v = f_emb._embed_one_cached(text, is_query=True)
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



def classify_one_multi(
    text: str,
    protos: Dict[str, np.ndarray],
    per_label_threshold: float = 0.4
) -> Dict:
    """Multi-label : conserve toutes les classes dont la similarité dépasse un seuil."""
    v = f_emb._embed_one_cached(text, is_query=True)
    sims = {lbl: float(vec @ v) for lbl, vec in protos.items()}
    kept = [lbl for lbl, s in sims.items() if s >= per_label_threshold]
    kept.sort(key=lambda k: sims[k], reverse=True)
    return {"labels": kept if kept else ["Autre"], "sims": sims}




# Calcule l’embedding du texte.
# Compare avec tous les prototypes via similarité cosinus (dot product, car vecteurs normalisés).
# Règles de décision :
# top_sim < threshold → label = "Autre".
# (top_sim - second_sim) < margin → label = "Autre" (texte ambigu).
# Retourne : {"label": ..., "confidence": score_top, "sims": {label: score}}.



if __name__ == "__main__":
    # Petit banc d'essai autonome pour les fonctions du module.
    # Exécuter:  python -m src.categorisation_echantillion.embeddings_proto
    import os
    import textwrap

    # Option: changer de modèle via la variable d'env EMB_MODEL_NAME
    model_name = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-base")
    print(f"[Init] Chargement du modèle: {model_name}")
    f_emb.load_model(model_name)

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
    calibrate = cal_emb.embedding(alpha=None)  # alpha adaptatif par classe
    protos = calibrate.build_prototypes(shots, label_defs=label_defs)

    print("[Calib] Calibration des hyperparamètres de rejet (threshold, margin)…")
    thr, mar = calibrate.calibrate_threshold(shots, label_defs=label_defs)
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
