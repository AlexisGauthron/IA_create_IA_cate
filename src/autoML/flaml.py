from flaml import AutoML
from sklearn.metrics import f1_score
from flaml import AutoML
import joblib

class autoMl_flaml:

    def __init__(self,Nom_dossier : str, X_train, X_test, y_train, y_test):
        self.Nom_dossier = Nom_dossier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.automl = None
        

    def flaml(self,task : str = "classification",metric : str = "f1",time_budget : int = 300, taille_max_modele : int = None, espace_ram_libre : int = 0):
        
        print("[INFO] Recherche Meilleur Modele\n")

        self.automl = AutoML()
        self.automl.fit(
            X_train=self.X_train, y_train=self.y_train,
            task=task,
            metric=metric,
            time_budget=time_budget,          # 5 min
            log_file_name=f"{self.Nom_dossier}/flaml.log",# <- journal détaillé des essais
            verbose=3,                 # logs console
            mem_thres=taille_max_modele,           # ~1.5 Go max pour un essai : 1.5e9
            free_mem_ratio=espace_ram_libre,       # garder 20% de RAM libre : 0.20
        )

        print("Meilleur algo :", self.automl.best_estimator)
        print("Meilleure config :", self.automl.best_config)
        print("Perte (proxy) :", self.automl.best_loss)


    def predict_test(self):
        print("[INFO] Test\n")
        pred = self.automl.predict(self.X_test)
        print("F1:", f1_score(self.y_test, pred),"\n")


    def enregistrement_model(self):
        # Enregistrement model
        print("[INFO] Enregistrement model\n")
        joblib.dump(self.automl.model, f"{self.Nom_dossier}/flaml_model.joblib")   # export

    def chargement_model(self):
        # Chargement model
        print("[INFO] Chargement model\n")
        model = joblib.load(f"{self.Nom_dossier}/flaml_model.joblib")           # import
        return model




