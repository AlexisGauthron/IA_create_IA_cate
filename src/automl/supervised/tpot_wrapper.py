from tpot import TPOTClassifier
from sklearn.metrics import f1_score, accuracy_score
import joblib

class autoMl_tpot:

    def __init__(self,Nom_dossier : str, X_train, X_test, y_train, y_test):
        self.Nom_dossier = Nom_dossier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None
        

    def tpot1(self, metric: str = "f1", time_budget: int = 1, n_jobs: int = 1):

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
            n_jobs=n_jobs
        )
        self.pred.fit(self.X_train, self.y_train)



    def predict_test(self):
        """Évalue le modèle sur le jeu de test.

        TPOT v1.x n'a pas de méthode score() directe.
        On utilise fitted_pipeline_ pour prédire et calculer les métriques.
        """
        print("[INFO] Test\n")
        if self.y_test is not None:
            # Utiliser le pipeline entraîné pour prédire
            y_pred = self.pred.fitted_pipeline_.predict(self.X_test)

            # Calculer les scores
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            acc = accuracy_score(self.y_test, y_pred)

            print(f"Accuracy: {acc:.4f}")
            print(f"F1 (weighted): {f1:.4f}\n")
            return f1
        else:
            print("Erreur : Jeux de test non labellisé, impossible d'émettre un score\n")
            return None


    def enregistrement_model(self):
        """Enregistre le modèle entraîné.

        TPOT v1.x n'a plus de méthode export().
        On sauvegarde directement le pipeline sklearn avec joblib.
        """
        print("[INFO] Enregistrement model\n")

        # Sauvegarder le pipeline sklearn entraîné
        joblib.dump(self.pred.fitted_pipeline_, f"{self.Nom_dossier}/tpot_best_pipeline.joblib")
        print(f"Modèle sauvegardé: {self.Nom_dossier}/tpot_best_pipeline.joblib")


    def chargement_model(self):
        # Chargement model
        print("[INFO] Chargement model\n")
        model = joblib.load(f"{self.Nom_dossier}/tpot_best_pipeline.joblib")
        return model



## Existance d'une librairie tpot2

