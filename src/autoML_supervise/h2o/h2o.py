import os
import json
from typing import Optional, Dict, Any, List

import pandas as pd
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")  # backend non interactif, aucune fenêtre ne s’ouvre
import matplotlib.pyplot as plt


class autoMl_h2o:
    """
    Wrapper structuré autour de H2O AutoML pour :
      - lancer des runs AutoML avec gestion de dossiers
      - sauvegarder le meilleur modèle + leaderboard
      - analyser le leader (perf, varimp, MOJO)
      - explorer les features des modèles du leaderboard
      - prédire sur le jeu de test avec sauvegarde des prédictions

    Paramètres principaux
    ---------------------
    :param project_root: chemin racine du projet (ex: "Resultats/MonProjet")
    :param X_train, X_test, y_train, y_test: données d'entraînement / test
    :param results_subdir: sous-dossier dans project_root pour H2O (ex: "h2o")
    :param auto_create_dirs: si True, crée automatiquement les dossiers
    """

    def __init__(
        self,
        project_root: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        *,
        results_subdir: str = "h2o",

        # ====== FLAGS DE SAUVEGARDE PAR DÉFAUT ======
        save_best_model: bool = True,
        save_leaderboard: bool = True,
        save_varimp_csv: bool = True,
        save_varimp_plot: bool = True,
        save_mojo: bool = True,
        save_features_csv: bool = True,
        save_features_json: bool = True,
        save_predictions: bool = True,
    ) -> None:
        
        # Données
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Chemins
        self.project_root = os.path.abspath(project_root)
        self.base_dir = os.path.join(self.project_root, results_subdir)
        self.current_run_name: Optional[str] = None
        self.current_run_dir: Optional[str] = None

        # Objets H2O / AutoML
        self.aml: Optional[H2OAutoML] = None
        self.test: Optional[h2o.H2OFrame] = None
        self.best_model_path: Optional[str] = None
        self.leaderboard_df: Optional[pd.DataFrame] = None
        self.perf_test = None # cache optionnel des performances sur test

        # ====== FLAGS DE SAUVEGARDE ======
        self.save_best_model_flag = save_best_model
        self.save_leaderboard_flag = save_leaderboard
        self.save_varimp_csv_flag = save_varimp_csv
        self.save_varimp_plot_flag = save_varimp_plot
        self.save_mojo_flag = save_mojo
        self.save_features_csv_flag = save_features_csv
        self.save_features_json_flag = save_features_json
        self.save_predictions_flag = save_predictions

        self._ensure_dir(self.base_dir)



    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------
    def _ensure_dir(self, path: str) -> None:
        """Crée le dossier s'il n'existe pas (idempotent)."""
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de créer le dossier '{path}': {e}")

    def _set_run_dir(self, run_name: str) -> None:
        """
        Définit le run courant (nom + dossier) à partir d'un run_name.
        Exemple: run_name = 'time_budget_600'
        """
        self.current_run_name = run_name
        self.current_run_dir = os.path.join(self.base_dir, run_name)
        self._ensure_dir(self.current_run_dir)

    def _get_best_model(self):
        """
        Récupère le modèle leader à partir de self.aml.
        Retourne None si indisponible (et loggue l'erreur).
        """
        if self.aml is None:
            print("[ERREUR] AutoML non entraîné (self.aml est None).")
            return None

        best_model = getattr(self.aml, "leader", None)
        if best_model is None:
            print("[ERREUR] Aucun 'leader' trouvé dans self.aml.")
            return None

        return best_model

    def _ensure_test_frame(self) -> bool:
        """
        S'assure que self.test est un H2OFrame prêt pour l'évaluation.
        Si self.test est None, le reconstruit à partir de X_test / y_test.
        Retourne True si OK, False en cas d'erreur.
        """
        if self.test is not None:
            return True

        if self.X_test is None or self.y_test is None:
            print("[ERREUR] X_test / y_test non définis, impossible de créer self.test.")
            return False

        try:
            print("[INFO] Reconstruction du H2OFrame de test...")
            test_df = pd.concat([self.X_test, self.y_test.rename("label")], axis=1)
        except Exception as e:
            print(f"[ERREUR] Impossible de concaténer X_test et y_test : {e}")
            return False

        try:
            self.test = h2o.H2OFrame(test_df)
            self.test["label"] = self.test["label"].asfactor()
            return True
        except Exception as e:
            print(f"[ERREUR] Impossible de créer le H2OFrame de test ou de caster 'label' en factor : {e}")
            return False

    # ------------------------------------------------------------------
    # Gestion du cluster H2O
    # ------------------------------------------------------------------
    def init_cluster(self, **init_kwargs) -> None:
        """
        Initialise le cluster H2O (wrapper sur h2o.init).

        :param init_kwargs: paramètres passés à h2o.init (ex: ip, port, nthreads, etc.)
        """
        print("[INFO] Initialisation du cluster H2O")
        h2o.init(**init_kwargs)

    def arreter_cluster(self, prompt: bool = False) -> None:
        """
        Arrête le cluster H2O proprement.

        :param prompt: si True, demande une confirmation (voir h2o.shutdown)
        """
        print("[INFO] Arrêt du cluster H2O")
        try:
            h2o.shutdown(prompt=prompt)
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur lors de l'arrêt du cluster H2O : {e}")

    # ------------------------------------------------------------------
    # Lancement d'un run AutoML
    # ------------------------------------------------------------------
    def start_automl(
        self,
        time_budget: int = 600,
        *,
        run_name: Optional[str] = None,
        export_checkpoints_subdir: str = "checkpoints",
        automl_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Lance un run AutoML H2O complet.

        - Crée un run_name (par défaut: f"time_budget_{time_budget}")
        - Crée le dossier de run
        - Convertit train/test en H2OFrame
        - Entraîne AutoML
        - Sauvegarde le meilleur modèle + leaderboard
        - Lance l'analyse du modèle + sauvegarde des features

        :param time_budget: temps max (en secondes) pour AutoML
        :param run_name: nom du run (dossier). Si None → "time_budget_{time_budget}"
        :param export_checkpoints_subdir: sous-dossier pour les checkpoints H2O
        :param automl_kwargs: autres paramètres à passer à H2OAutoML
        """
        if run_name is None:
            run_name = f"time_budget_{time_budget}"

        self._set_run_dir(run_name)

        # (Re)init cluster si besoin
        self.init_cluster()

        print(f"[INFO] Démarrage H2O AutoML (run: {run_name})")

        # DataFrames pandas → H2OFrame
        train_df = pd.concat([self.X_train, self.y_train.rename("label")], axis=1)
        test_df = pd.concat([self.X_test, self.y_test.rename("label")], axis=1)

        train = h2o.H2OFrame(train_df)
        self.test = h2o.H2OFrame(test_df)

        y = "label"
        x = train.columns
        x.remove(y)

        train[y] = train[y].asfactor()
        self.test[y] = self.test[y].asfactor()

        # Construction des kwargs AutoML
        if automl_kwargs is None:
            automl_kwargs = {}

        export_checkpoints_dir = os.path.join(self.current_run_dir, export_checkpoints_subdir)
        self._ensure_dir(export_checkpoints_dir)

        self.aml = H2OAutoML(
            max_runtime_secs=time_budget,
            project_name=run_name,
            seed=42,
            verbosity="info",
            export_checkpoints_dir=export_checkpoints_dir,
            **automl_kwargs,
        )
        self.aml.train(x=x, y=y, training_frame=train)

        # Sauvegarde du leader + leaderboard
        if self.save_best_model_flag == True:
            self.save_best_model()
        if self.save_leaderboard_flag == True:
            self.save_leaderboard()

        # Analyse + features de tous les modèles
        self.analyser_modele()
        print("###################### Fin analyse ! ####################\n")
        self.sauvegarder_features_tous_modeles()



    # ------------------------------------------------------------------
    # Sauvegarde / récupération des fichiers principaux
    # ------------------------------------------------------------------
    def save_best_model(self, subdir: str = "best") -> None:
        """
        Sauvegarde le meilleur modèle (leader) dans un sous-dossier du run courant.

        :param subdir: sous-dossier dans current_run_dir (ex: "best")
        """
        best_model = self._get_best_model()
        if best_model is None or self.current_run_dir is None:
            return

        dest_dir = os.path.join(self.current_run_dir, subdir)
        self._ensure_dir(dest_dir)

        try:
            self.best_model_path = h2o.save_model(
                model=best_model,
                path=dest_dir,
                force=True,
            )
            print(f"[SAVED] Meilleur modèle sauvegardé ici → {self.best_model_path}")
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de sauvegarder le meilleur modèle : {e}")


    def save_leaderboard(self, filename: str = "leaderboard.csv") -> None:
        """
        Sauvegarde la leaderboard au format CSV dans le run courant + garde une copie en mémoire (pandas).

        :param filename: nom du fichier CSV (ex: 'leaderboard.csv')
        """
        if self.aml is None or self.current_run_dir is None:
            print("[ERREUR] Impossible de sauvegarder la leaderboard (aucun run courant).")
            return

        try:
            lb = get_leaderboard(self.aml, extra_columns="ALL")
            self.leaderboard_df = lb.as_data_frame()
        except Exception as e:
            print(f"[ERREUR] Impossible de récupérer la leaderboard AutoML : {e}")
            return

        csv_path = os.path.join(self.current_run_dir, filename)
        try:
            h2o.download_csv(lb, csv_path)
            print(f"[SAVED] Leaderboard → {csv_path}")
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de sauvegarder la leaderboard en CSV : {e}")

        print(self.leaderboard_df.head(10))

    def list_runs(self) -> List[str]:
        """
        Liste les sous-dossiers de base_dir (chaque sous-dossier est un run possible).
        """
        if not os.path.isdir(self.base_dir):
            return []
        return sorted(
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        )



    # ------------------------------------------------------------------
    # Chargement d'un modèle existant
    # ------------------------------------------------------------------
    def charger_modele_existant(
        self,
        model_path: Optional[str] = None,
        *,
        run_name: Optional[str] = None,
        search_prefixes: Optional[List[str]] = None,
    ) -> None:
        """
        Charge un modèle déjà entraîné comme 'leader' (pour réanalyse / prédiction).

        Deux modes :
        - mode direct : model_path donné (fichier .zip ou id H2O sauvegardé)
        - mode recherche :
            - on choisit un run_name (ou le dernier run)
            - on cherche un fichier commençant par un des préfixes dans ce dossier

        :param model_path: chemin complet du modèle sauvegardé (prioritaire si non None)
        :param run_name: nom du run dans base_dir où chercher le modèle
        :param search_prefixes: liste de préfixes d'algos (ex: ["StackedEnsemble", "XGBoost", "GBM"])
        """
        self.init_cluster()

        if model_path is not None:
            # Chargement direct
            print(f"[INFO] Chargement du modèle à partir de : {model_path}")
            loaded_model = h2o.load_model(model_path)
            self.aml = type("obj", (object,), {})()  # faux objet pour avoir .leader
            self.aml.leader = loaded_model
            self.best_model_path = model_path
            print("[INFO] Modèle chargé avec succès !")
            print("ID du modèle :", self.aml.leader.model_id)
            return

        # Mode recherche
        if run_name is None:
            # On prend le dernier run (ordre alphabétique)
            runs = self.list_runs()
            if not runs:
                raise FileNotFoundError(f"Aucun run trouvé dans {self.base_dir}")
            run_name = runs[-1]

        self._set_run_dir(run_name)
        if self.current_run_dir is None:
            raise FileNotFoundError("Aucun dossier de run courant valide.")

        if search_prefixes is None:
            search_prefixes = ["StackedEnsemble", "XGBoost", "GBM", "DRF", "GLM"]

        fichiers = os.listdir(self.current_run_dir)
        candidates = [
            f for f in fichiers
            if any(f.startswith(pref) for pref in search_prefixes)
        ]

        if not candidates:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {self.current_run_dir}")

        # On prend le plus récent (ordre alphabétique ou autre logique)
        candidate = sorted(candidates)[-1]
        chemin_modele = os.path.join(self.current_run_dir, candidate)

        print(f"[INFO] Chargement du modèle : {chemin_modele}")
        loaded_model = h2o.load_model(chemin_modele)
        self.aml = type("obj", (object,), {})()
        self.aml.leader = loaded_model
        self.best_model_path = chemin_modele

        print("[INFO] Modèle chargé avec succès !")
        print("ID du modèle :", self.aml.leader.model_id)



    # ------------------------------------------------------------------
    # Analyse du modèle leader
    # ------------------------------------------------------------------
    def analyser_modele(self) -> Dict[str, Any]:
        """
        Analyse le meilleur modèle H2O AutoML :

        - Performance sur le test (si self.test dispo)
        - Importance des variables (+ export CSV)
        - Plot d'importance sauvegardé en PNG
        - Export MOJO

        Retourne un dict rassemblant ce qui a pu être calculé.
        """
        resultats = {
            "perf_test": None,
            "var_importance": None,
            "vi_path": None,
            "mojo_path": None,
            "varimp_plot_path": None,
        }

        best_model = self._get_best_model()
        if best_model is None:
            return resultats

        # 1) Perf sur test (indépendant des flags de sauvegarde)
        if self._ensure_test_frame():
            try:
                perf = best_model.model_performance(self.test)
                print("\n=== Performances sur le jeu de test ===")
                print(perf)
                self.perf_test = perf
                resultats["perf_test"] = perf
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de calculer la performance sur le test : {e}")
        else:
            print("[INFO] self.test indisponible → pas de performance sur le test.")

        if self.current_run_dir is None:
            print("[AVERTISSEMENT] current_run_dir est None, certains fichiers ne pourront pas être sauvegardés.")
            return resultats

        # 2) Importance des variables + CSV
        try:
            vi = best_model.varimp(use_pandas=True)
            if vi is not None:
                print("\n=== Top 20 variables les plus importantes ===")
                print(vi.head(20))
                resultats["var_importance"] = vi

                if self.save_varimp_csv_flag:
                    try:
                        vi_path = os.path.join(self.current_run_dir, "variable_importance.csv")
                        vi.to_csv(vi_path, index=False)
                        resultats["vi_path"] = vi_path
                        print(f"[INFO] Importance des variables sauvegardée dans : {vi_path}")
                    except Exception as e:
                        print(f"[AVERTISSEMENT] Impossible de sauvegarder l'importance des variables : {e}")
            else:
                print("[INFO] Aucune importance de variable disponible (vi est None).")
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de récupérer l'importance des variables : {e}")

        # 3) Plot varimp → PNG
        if self.save_varimp_plot_flag:
            print("\n[DEBUG] Génération du varimp_plot (sans blocage)...")
            try:
                best_model.varimp_plot(num_of_features=20)
                plot_path = os.path.join(self.current_run_dir, "varimp_plot.png")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                resultats["varimp_plot_path"] = plot_path
                print(f"[DEBUG] varimp_plot sauvegardé dans : {plot_path}")
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de générer le varimp_plot : {e}")

        # 4) MOJO
        if self.save_mojo_flag:
            try:
                mojo_path = best_model.download_mojo(
                    path=self.current_run_dir,
                    get_genmodel_jar=True,
                )
                resultats["mojo_path"] = mojo_path
                print(f"[INFO] MOJO prêt pour la production → {mojo_path}")
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de télécharger le MOJO : {e}")

        return resultats

    # ------------------------------------------------------------------
    # Exploration des features de tous les modèles
    # ------------------------------------------------------------------
    def sauvegarder_features_tous_modeles(self) -> Dict[str, Any]:
        """
        Sauvegarde, pour TOUS les modèles du leaderboard AutoML :
        - les features internes utilisées
        - les noms originaux (si dispo)
        - les colonnes ignorées

        Résultats :
        - CSV: all_models_features.csv (listes converties en chaînes)
        - JSON: all_models_features.json (listes intactes)
        """
        resultats = {
            "df_features": None,
            "csv_path": None,
            "json_path": None,
            "nb_models_total": 0,
            "nb_models_ok": 0,
            "failed_models": [],
        }

        best_model = self._get_best_model()
        if best_model is None or self.current_run_dir is None:
            return resultats

        # Leaderboard (réutilise le cache si dispo)
        if self.leaderboard_df is not None:
            lb_df = self.leaderboard_df
        else:
            try:
                lb = get_leaderboard(self.aml, extra_columns="ALL")
                if lb is None:
                    print("[ERREUR] get_leaderboard a retourné None.")
                    return resultats
                lb_df = lb.as_data_frame()
                self.leaderboard_df = lb_df
            except Exception as e:
                print(f"[ERREUR] Impossible de récupérer le leaderboard AutoML : {e}")
                return resultats

        if "model_id" not in lb_df.columns:
            print("[ERREUR] La leaderboard ne contient pas de colonne 'model_id'.")
            return resultats

        model_ids = lb_df["model_id"].tolist()
        resultats["nb_models_total"] = len(model_ids)

        if not model_ids:
            print("[INFO] Aucun modèle dans le leaderboard.")
            return resultats

        rows = []
        failed_models = []

        for mid in tqdm(model_ids, desc="Sauvegarde des features modèles", unit="modèle"):
            try:
                m = h2o.get_model(mid)
            except Exception as e:
                print(f"\n[WARN] Impossible de charger le modèle {mid}: {e}")
                failed_models.append({"model_id": mid, "error": str(e)})
                continue

            try:
                output = m._model_json.get("output", {}) or {}

                names = output.get("names", []) or []
                if not isinstance(names, (list, tuple)):
                    names = [str(names)]

                orig_names = output.get("original_names", None)
                if orig_names is not None and not isinstance(orig_names, (list, tuple)):
                    orig_names = [str(orig_names)]

                ignored = m.actual_params.get("ignored_columns", []) or []
                if not isinstance(ignored, (list, tuple)):
                    ignored = [str(ignored)]

                try:
                    rank = int(lb_df.index[lb_df["model_id"] == mid][0]) + 1
                except Exception:
                    rank = None

                row = {
                    "model_id": mid,
                    "algo": getattr(m, "algo", None),
                    "leaderboard_rank": rank,
                    "used_features": list(names),
                    "original_features": orig_names,
                    "ignored_columns": list(ignored),
                }
                rows.append(row)

            except Exception as e:
                print(f"\n[WARN] Problème lors de l'extraction des features pour {mid}: {e}")
                failed_models.append({"model_id": mid, "error": str(e)})
                continue

        if not rows:
            print("[ERREUR] Aucune feature n'a pu être extraite pour les modèles.")
            resultats["failed_models"] = failed_models
            return resultats

        df_features = pd.DataFrame(rows)
        resultats["df_features"] = df_features
        resultats["nb_models_ok"] = len(rows)
        resultats["failed_models"] = failed_models

        # Sauvegarde CSV
        if self.save_features_csv_flag:
            csv_path = os.path.join(self.current_run_dir, "all_models_features.csv")
            try:
                df_csv = df_features.copy()
                for col in ["used_features", "original_features", "ignored_columns"]:
                    if col in df_csv.columns:
                        df_csv[col] = df_csv[col].apply(
                            lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
                        )
                df_csv.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"\n[SAVED] Features de tous les modèles → {csv_path}")
                resultats["csv_path"] = csv_path
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de sauvegarder le CSV des features : {e}")

        # Sauvegarde JSON
        if self.save_features_json_flag:
            json_path = os.path.join(self.current_run_dir, "all_models_features.json")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
                print(f"[SAVED] Détail features (JSON) → {json_path}")
                resultats["json_path"] = json_path
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de sauvegarder le JSON des features : {e}")

        return resultats

    # ------------------------------------------------------------------
    # Prédictions sur le jeu de test
    # ------------------------------------------------------------------
    def predict_test(
        self,
        *,
        filename: str = "predictions_test.csv",
    ):
        """
        Utilise le modèle leader pour prédire sur le jeu de test et :

        - Affiche les performances (si calculables)
        - Affiche éventuellement la matrice de confusion (classification)
        - Sauvegarde un CSV combinant :
            X_test + y_test (label) + colonnes de prédiction H2O

        :param save_preds: si True, sauvegarde le CSV
        :param filename: nom du fichier CSV (dans current_run_dir)
        :return: objet 'perf' (model_performance) ou None
        """
        best_model = self._get_best_model()
        if best_model is None:
            return None

        if not self._ensure_test_frame():
            return None

        perf = None
        print(f"[INFO] Évaluation du modèle leader : {best_model.model_id}")

        # Performance
        try:
            perf = best_model.model_performance(self.test)
            print("\n=== Performance sur le jeu de test ===")
            print(perf)
            self.perf_test = perf
        except Exception as e:
            print(f"[ERREUR] Impossible de calculer la performance sur le jeu de test : {e}")
        else:
            # Matrice de confusion si dispo
            try:
                cm = perf.confusion_matrix()
                if cm is not None:
                    print("\n=== Matrice de confusion (test) ===")
                    print(cm)
            except Exception as e:
                print(f"[INFO] Pas de matrice de confusion disponible ou erreur lors de son calcul : {e}")

        # Prédictions détaillées
        try:
            preds = best_model.predict(self.test)
        except Exception as e:
            print(f"[ERREUR] Impossible de générer les prédictions sur le jeu de test : {e}")
            return perf

        if self.save_predictions_flag and self.current_run_dir is not None:
            try:
                test_pd = self.test.as_data_frame()
                preds_pd = preds.as_data_frame()
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de convertir H2OFrame en DataFrame pandas : {e}")
            else:
                try:
                    merged = pd.concat(
                        [test_pd.reset_index(drop=True),
                         preds_pd.reset_index(drop=True)],
                        axis=1,
                    )
                    preds_path = os.path.join(self.current_run_dir, filename)
                    merged.to_csv(preds_path, index=False, encoding="utf-8")
                    print(f"\n[SAVED] Prédictions complètes (test) → {preds_path}")
                except Exception as e:
                    print(f"[AVERTISSEMENT] Impossible de sauvegarder les prédictions dans un CSV : {e}")

        return perf

    # ------------------------------------------------------------------
    # Pipeline complet: run + prédiction + arrêt cluster
    # ------------------------------------------------------------------
    def use_all(
        self,
        time_budget: int,
        *,
        run_name: Optional[str] = None,
        save_preds: bool = True,
        shutdown_after: bool = True,
    ):
        """
        Pipeline "one-shot" pour :
          1) lancer AutoML
          2) analyser le leader
          3) faire les prédictions sur le test
          4) arrêter le cluster (optionnel)

        :param time_budget: temps max (en secondes) pour AutoML
        :param run_name: nom explicite du run (sinon auto)
        :param save_preds: si True, sauvegarde les prédictions test
        :param shutdown_after: si True, arrête le cluster H2O à la fin
        """
        self.start_automl(time_budget=time_budget, run_name=run_name)
        print("[DEBUG] Début évaluation sur le jeu de test\n")
        score = self.predict_test()
        print("[DEBUG] Fin évaluation sur le jeu de test\n")

        if shutdown_after:
            self.arreter_cluster()

        return score
