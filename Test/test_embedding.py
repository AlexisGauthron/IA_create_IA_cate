
import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


# Pour chaque prototype, on garde les labels dont la similarité dépasse le seuil.
# Permet d’attribuer plusieurs labels à un texte.
# Si aucun label ne passe le seuil → "Autre".


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

# --------- Démo multi-label ----------
multi_tests = [
    "Le colis est perdu et j'ai besoin d'un avoir.",
    "Après la démo, pouvez-vous m'envoyer le devis ?",
    "Impossible de me connecter pour récupérer ma facture.",
    "Je dois modifier l'adresse de livraison et connaître le prix.",
]



if __name__ == "__main__":
    # Petit banc d'essai autonome pour les fonctions du module.

    import src.few_shot.embedding.few_shot as few_shot_module
    exp = few_shot_module.FewShotExperiment(shots, tests, model_name="intfloat/multilingual-e5-base")

    # 1) Un run unique (garde tous les prints)
    thr, mar, acc = exp.run_once(
        mono_label=True, multi_label=False,
        allow_defs=False, label_defs=label_defs,        # ou allow_defs=True pour génération LLM
        alpha_def=None, alpha_base=0.30, alpha_max_extra=0.40, alpha_lam=6,
        perc=10, class_balanced=True,
        thr_bounds=(0.20, 0.60), mar_bounds=(0.02, 0.15),
    )

    # 2) Un sweep de configs (affiche un header pour chaque combo + tes logs habituels)
    # grid_alpha = {"alpha_def":[None, 0.2], "alpha_base":[0.25, 0.30],
    #             "alpha_max_extra":[0.30, 0.40], "alpha_lam":[4, 6]}
    # grid_calib = {"perc":[5, 10], "class_balanced":[True],
    #             "thr_bounds":[(0.2,0.6)], "mar_bounds":[(0.02,0.15)]}

    # exp.sweep(grid_alpha, grid_calib, defs_kwargs={"allow_defs": False, "label_defs": label_defs})
    # print("\n=== Résumé final ===\n\n")
    # protos, thr, mar, cfg = exp.get_prototypes(which="best")
    
    # print("\n=== Résumé final ===")
    # print(f"Meilleure config : {cfg}")

    # exp.print_all_results()

