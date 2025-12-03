import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score
import joblib

class autoMl_autogluon:

    def __init__(self,Nom_dossier : str, X_train, X_test, y_train, y_test):
        self.ag_dossier = f"{Nom_dossier}/ag_out"
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None
        

    def autogluon(self,presets : str = "medium_quality_faster_train",metric : str = "auto",time_budget : int = 60):

        print("[INFO] Recherche Meilleur Modele autogluon\n")

        # Convertir en DataFrame si c'est un numpy array (après StandardScaler)
        if not isinstance(self.X_train, pd.DataFrame):
            train_df = pd.DataFrame(self.X_train)
        else:
            train_df = self.X_train.copy()

        # Gestion de X_test = None (pas de jeu de test fourni)
        if self.X_test is not None:
            if not isinstance(self.X_test, pd.DataFrame):
                test_df = pd.DataFrame(self.X_test)
            else:
                test_df = self.X_test.copy()
            if self.y_test is not None:
                test_df["label"] = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        else:
            test_df = None
            print("[INFO] Pas de jeu de test fourni - utilisation des scores de validation")

        train_df["label"] = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train

        # Détection automatique de la métrique selon le type de problème
        if metric == "auto":
            n_classes = train_df["label"].nunique()
            if n_classes == 2:
                metric = "f1"
                print(f"[INFO] Problème binaire détecté -> métrique: f1")
            else:
                metric = "f1_macro"
                print(f"[INFO] Problème multiclasse détecté ({n_classes} classes) -> métrique: f1_macro")

        ag_dossier = f"{self.ag_dossier}/ag_out"

        self.pred = TabularPredictor(label="label", path=ag_dossier, verbosity=2, eval_metric=metric).fit(
            train_df,
            time_limit=time_budget,
            presets=presets
        )

        # Leaderboard (progrès et scores par modèle)
        # Si test_df existe et a des labels, on peut calculer des scores sur le test set
        if test_df is not None and self.y_test is not None:
            lb = self.pred.leaderboard(test_df, silent=True)
            print(lb.head(10))
        else:
            # Leaderboard sans données test (scores de validation seulement)
            lb = self.pred.leaderboard(silent=True)
            print("[INFO] Scores de validation uniquement (pas de jeu de test)")
            print(lb.head(10))

        # Résumé - fit_summary() désactivé car incompatibilité NumPy 2.0 / Bokeh
        # print(self.pred.fit_summary())  # Cause: AttributeError: module 'numpy' has no attribute 'bool8'
        print(f"[INFO] Modèles sauvegardés dans: {ag_dossier}")


    def predict_test(self):
        print("[INFO] Test\n")

        # Utiliser self.pred (le TabularPredictor entraîné), pas self.automl
        if self.pred is None:
            print("[ERREUR] Modèle non entraîné - appeler autogluon() d'abord\n")
            return None

        # Si X_test est None ou y_test est None, retourner le score de validation
        if self.X_test is None or self.y_test is None:
            print("[INFO] Pas de jeu de test disponible ou pas de labels")
            print("[INFO] Retour du score de validation du meilleur modèle")
            # Retourner le score de validation du meilleur modèle via leaderboard
            lb = self.pred.leaderboard(silent=True)
            best_score = lb.iloc[0]['score_val']  # Premier modèle = meilleur
            best_model = lb.iloc[0]['model']
            print(f"[INFO] Meilleur modèle: {best_model}")
            print(f"[INFO] Score de validation: {best_score:.4f}\n")
            return best_score

        # Convertir X_test en DataFrame si c'est un numpy array (après StandardScaler)
        X_test_df = pd.DataFrame(self.X_test) if not isinstance(self.X_test, pd.DataFrame) else self.X_test
        predictions = self.pred.predict(X_test_df)

        # Adapter le calcul F1 selon le nombre de classes
        y_test_values = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        n_classes = len(set(y_test_values))
        if n_classes == 2:
            score = f1_score(self.y_test, predictions)
        else:
            score = f1_score(self.y_test, predictions, average='macro')
        print(f"F1{'_macro' if n_classes > 2 else ''}: {score}\n")
        return score
    

    def chargement_model(self):
        # Chargement model
        print("[INFO] Chargement model\n")
        from autogluon.tabular import TabularPredictor
        model = TabularPredictor.load(self.ag_dossier)
        return model
