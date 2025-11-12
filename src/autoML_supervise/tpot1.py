from tpot import TPOTClassifier
from sklearn.metrics import f1_score
import joblib

class autoMl_tpot:

    def __init__(self,Nom_dossier : str, X_train, X_test, y_train, y_test):
        self.Nom_dossier = Nom_dossier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None
        

    def tpot1(self,metric : str = "f1",time_budget : int = 5):
        
        print("[INFO] Recherche Meilleur Modele tpot\n")

        self.pred = TPOTClassifier(
            generations=7,
            population_size=25,
            scoring=metric,
            max_time_mins=time_budget,
            cv=5,
            verbosity=3,                          # logs d’avancement
            periodic_checkpoint_folder="tpot_ckpt", # <- checkpoints réguliers
            random_state=42,
            n_jobs=-1
        )
        self.pred.fit(self.X_train, self.y_train)



    def predict_test(self):
        print("[INFO] Test\n")
        score = self.pred.score(self.X_test, self.y_test)
        print("F1:", score,"\n")

        return score


    def enregistrement_model(self):
        # Enregistrement model
        print("[INFO] Enregistrement model\n")
        self.pred.export(f"{self.Nom_dossier}/tpot_best_pipeline.py")      # pipeline sklearn reproductible

        # 2) Objet entraîné (pipeline sklearn)
        joblib.dump(self.pred.fitted_pipeline_, f"{self.Nom_dossier}/tpot_best_pipeline.joblib")


    def chargement_model(self):
        # Chargement model
        print("[INFO] Chargement model\n")
        model = joblib.load(f"{self.Nom_dossier}/tpot_best_pipeline.joblib")
        return model



## Existance d'une librairie tpot2

