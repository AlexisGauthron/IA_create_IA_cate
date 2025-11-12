import pandas as pd
import h2o
from h2o.automl import H2OAutoML

class autoMl_h2o:

    def __init__(self,Nom_dossier : str, X_train, X_test, y_train, y_test):
        self.Nom_dossier = Nom_dossier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.aml = None

        self.test = None
        

    def h2o(self, time_budget : int = 300):
        
        print("[INFO] Recherche Meilleur Modele h2o\n")

        # 0) Démarrer H2O
        h2o.init()  # éventuellement: max_mem_size="6G"

        # 1) Reconstituer un DF d'entraînement avec la cible
        train_df = pd.concat([self.X_train, self.y_train.rename("label")], axis=1)
        test_df  = pd.concat([self.X_test,  self.y_test.rename("label")],  axis=1)

        # 2) Convertir en H2OFrame
        train = h2o.H2OFrame(train_df)
        self.test  = h2o.H2OFrame(test_df)

        # 3) Définir y/x (x doit être une LISTE) + typage cible (classification)
        y = "label"
        x = [c for c in train.columns if c != y]
        train[y] = train[y].asfactor()
        self.test[y]  = self.test[y].asfactor()

        # 4) Lancer AutoML
        self.aml = H2OAutoML(
            max_runtime_secs=time_budget,
            project_name="demo_automl",
            seed=42,
            verbosity="info",
        )
        self.aml.train(x=x, y=y, training_frame=train)

        # 5) Résultats
        print(self.aml.leaderboard.head(rows=10))


    def predict_test(self):

        print("[INFO] Test\n")
        perf = self.aml.leader.model_performance(self.test)
        print("Resultat :\n",perf)
        return perf


    def chargement_model(self):
        # Chargement model
        print("[INFO] Chargement model\n")
        from autogluon.tabular import TabularPredictor
        model = TabularPredictor.load(self.ag_dossier)
        return model
    




