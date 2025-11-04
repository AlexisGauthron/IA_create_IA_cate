import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score
import joblib

class autoMl_flaml:

    def __init__(self,Nom_dossier : str, X_train, X_test, y_train, y_test):
        self.ag_dossier = f"{Nom_dossier}/ag_out"
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None
        

    def flaml(self,task : str = "classification",metric : str = "f1",time_budget : int = 300, taille_max_modele : int = None, espace_ram_libre : int = 0):
        
        print("[INFO] Recherche Meilleur Modele\n")

        train_df = self.X_train.copy(); train_df["label"] = self.y_train.values
        test_df  = self.X_test.copy(); test_df["label"]  = self.y_test.values

        ag_dossier = f"{self.ag_dossier}/ag_out"

        pred = TabularPredictor(label="label", path=ag_dossier, verbosity=2).fit(
            train_df,
            time_limit=300,                       # 5 min
            presets="medium_quality_faster_train" # compromis vitesse/qualité
        )

        # Leaderboard (progrès et scores par modèle)
        lb = pred.leaderboard(test_df, silent=True, extra_info=True)
        print(lb.head(10))

        # Résumé + où sont stockés les checkpoints (ag_out/)
        print(pred.fit_summary())


    def predict_test(self):
        from sklearn.metrics import f1_score

        print("[INFO] Test\n")
        pred = self.pred.predict(self.X_test)
        print("F1:", f1_score(self.y_test, pred),"\n")

    def chargement_model(self):
        # Chargement model
        print("[INFO] Chargement model\n")
        from autogluon.tabular import TabularPredictor
        model = TabularPredictor.load(self.ag_dossier)
        return model
