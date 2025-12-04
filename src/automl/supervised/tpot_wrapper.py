"""
Wrapper pour TPOT AutoML.
"""

import joblib
from sklearn.metrics import accuracy_score, f1_score
from tpot import TPOTClassifier

# =============================================================================
# Classe TPOTWrapper
# =============================================================================
# Renommée de 'autoMl_tpot' vers 'TPOTWrapper'
# Raison: 'autoMl_tpot' mélangeait camelCase et snake_case
#         Les classes doivent être en PascalCase
#         TPOT est un acronyme → tout en majuscules
# =============================================================================


class TPOTWrapper:
    """
    Wrapper pour le framework TPOT (Tree-based Pipeline Optimization Tool).

    Exemple:
        wrapper = TPOTWrapper(
            output_dir="outputs/projet",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        wrapper.tpot1(time_budget=5)
        score = wrapper.predict_test()
    """

    def __init__(
        self,
        output_dir: str,  # Renommé de 'Nom_dossier' → snake_case anglais
        X_train,
        X_test,
        y_train,
        y_test,
    ):
        """
        Initialise le wrapper TPOT.

        Args:
            output_dir: Dossier de sortie pour les modèles
                        (anciennement 'Nom_dossier')
            X_train: Features d'entraînement
            X_test: Features de test
            y_train: Cible d'entraînement
            y_test: Cible de test
        """
        self.output_dir = output_dir  # Renommé de 'Nom_dossier'
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None

    def tpot1(self, metric: str = "f1", time_budget: int = 1, n_jobs: int = 1):
        """
        Lance l'entraînement TPOT.

        Args:
            metric: Métrique d'optimisation
            time_budget: Budget temps en minutes
            n_jobs: Nombre de jobs parallèles (1 par défaut pour éviter problèmes Dask)
        """
        print("[INFO] Recherche Meilleur Modele tpot\n")

        # TPOT v1.x utilise scorers au lieu de scoring
        # n_jobs=1 par défaut pour éviter les problèmes Dask "No valid workers"
        self.pred = TPOTClassifier(
            generations=7,
            population_size=25,
            scorers=[metric],
            scorers_weights=[1],
            max_time_mins=time_budget,
            cv=5,
            verbose=3,
            random_state=42,
            n_jobs=n_jobs,
        )
        self.pred.fit(self.X_train, self.y_train)

    def predict_test(self):
        """
        Évalue le modèle sur le jeu de test.

        TPOT v1.x n'a pas de méthode score() directe.
        On utilise fitted_pipeline_ pour prédire et calculer les métriques.
        """
        print("[INFO] Test\n")
        if self.y_test is not None:
            # Utiliser le pipeline entraîné pour prédire
            y_pred = self.pred.fitted_pipeline_.predict(self.X_test)

            # Calculer les scores
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            acc = accuracy_score(self.y_test, y_pred)

            print(f"Accuracy: {acc:.4f}")
            print(f"F1 (weighted): {f1:.4f}\n")
            return f1
        else:
            print("Erreur : Jeux de test non labellisé, impossible d'émettre un score\n")
            return None

    def enregistrement_model(self):
        """
        Enregistre le modèle entraîné.

        TPOT v1.x n'a plus de méthode export().
        On sauvegarde directement le pipeline sklearn avec joblib.
        """
        print("[INFO] Enregistrement model\n")

        # Sauvegarder le pipeline sklearn entraîné
        joblib.dump(self.pred.fitted_pipeline_, f"{self.output_dir}/tpot_best_pipeline.joblib")
        print(f"Modèle sauvegardé: {self.output_dir}/tpot_best_pipeline.joblib")

    def chargement_model(self):
        """Charge un modèle sauvegardé."""
        print("[INFO] Chargement model\n")
        model = joblib.load(f"{self.output_dir}/tpot_best_pipeline.joblib")
        return model


# Note: Il existe aussi une librairie tpot2
