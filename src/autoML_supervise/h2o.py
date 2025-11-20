import pandas as pd
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import os

class autoMl_h2o:
    def __init__(self, Nom_dossier: str, X_train, X_test, y_train, y_test):
        self.Nom_dossier = f"{Nom_dossier}/h2o"
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.aml = None
        self.test = None
        self.best_model_path = None  # ← on va stocker le chemin du modèle sauvegardé

        # Créer le dossier si n'existe pas
        os.makedirs(self.Nom_dossier, exist_ok=True)

    def h2o(self, time_budget: int = 600):
        print("[INFO] Démarrage H2O AutoML\n")
        h2o.init()

        train_df = pd.concat([self.X_train, self.y_train.rename("label")], axis=1)
        test_df  = pd.concat([self.X_test,  self.y_test.rename("label")],  axis=1)

        train = h2o.H2OFrame(train_df)
        self.test = h2o.H2OFrame(test_df)

        y = "label"
        x = train.columns
        x.remove(y)
        train[y] = train[y].asfactor()
        self.test[y] = self.test[y].asfactor()

        self.aml = H2OAutoML(
            max_runtime_secs=time_budget,
            project_name="mon_projet",
            seed=42,
            verbosity="info",
            export_checkpoints_dir=f"{self.Nom_dossier}/modeles",  # ← sauvegarde automatique des checkpoints
        )
        self.aml.train(x=x, y=y, training_frame=train)

        # === SAUVEGARDE DU MEILLEUR MODÈLE ===
        self.best_model_path = h2o.save_model(
            model=self.aml.leader,
            path=self.Nom_dossier,
            force=True
        )
        print(f"[SAVED] Meilleur modèle sauvegardé ici → {self.best_model_path}")

        # Sauvegarde aussi le leaderboard au format CSV (pratique)
        lb = get_leaderboard(self.aml, extra_columns='ALL')
        lb_path = os.path.join(self.Nom_dossier, "leaderboard.csv")
        h2o.download_csv(lb, lb_path)
        print(f"[SAVED] Leaderboard → {lb_path}")

        print(self.aml.leaderboard.head(10))

    # =================================================================
    # NOUVELLE MÉTHODE : charger un modèle déjà entraîné
    # =================================================================
    def charger_modele_existant(self):
        print("[INFO] Chargement du modèle H2O déjà entraîné\n")
        h2o.init()  # redémarre le cluster si besoin

        # On cherche automatiquement le modèle dans le dossier
        modeles_trouves = [f for f in os.listdir(self.Nom_dossier) if f.startswith("StackedEnsemble") or f.startswith("XGBoost") or f.startswith("GBM")]
        if not modeles_trouves:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {self.Nom_dossier}")

        # On prend le plus récent (ou tu peux spécifier le nom exact)
        chemin_modele = os.path.join(self.Nom_dossier, sorted(modeles_trouves)[-1])
        print(f"Chargement du modèle : {chemin_modele}")

        self.aml = type('obj', (object,), {})()  # faux objet juste pour avoir .leader
        self.aml.leader = h2o.load_model(chemin_modele)
        self.best_model_path = chemin_modele

        print("Modèle chargé avec succès !")
        print("ID du modèle :", self.aml.leader.model_id)

    # =================================================================
    # Méthode pour tout avoir après chargement ou entraînement
    # =================================================================
    def analyser_modele(self):
        if self.aml is None or self.aml.leader is None:
            raise ValueError("Entraîne ou charge d'abord un modèle !")

        best_model = self.aml.leader

        # 1. Performance sur test
        if self.test is not None:
            perf = best_model.model_performance(self.test)
            print(perf)

        # 2. Importance des variables
        vi = best_model.varimp(use_pandas=True)
        if vi is not None:
            print("\n=== Top 20 variables les plus importantes ===")
            print(vi.head(20))
            vi_path = os.path.join(self.Nom_dossier, "variable_importance.csv")
            vi.to_csv(vi_path, index=False)
            print(f"Sauvegardé dans {vi_path}")

        # 3. Plot
        best_model.varimp_plot(num_of_features=20)

        # 4. Export MOJO (le top pour la prod)
        mojo_path = best_model.download_mojo(path=self.Nom_dossier, get_genmodel_jar=True)
        print(f"MOJO prêt pour la production → {mojo_path}")

        