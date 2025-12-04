import json
import os
import sys
from typing import Any

import h2o
import pandas as pd
from h2o.automl import H2OAutoML, get_leaderboard
from tqdm.auto import tqdm

# Force le flush automatique de stdout pour voir les logs en temps réel
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import matplotlib

matplotlib.use("Agg")  # backend non interactif, aucune fenêtre ne s’ouvre
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# =============================================================================
# Classe H2OWrapper
# =============================================================================
# Renommée de 'autoMl_h2o' vers 'H2OWrapper'
# Raison: 'autoMl_h2o' mélangeait camelCase et snake_case
#         Les classes doivent être en PascalCase
#         H2O est un nom propre → on garde les majuscules
# =============================================================================


class H2OWrapper:
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
        self.current_run_name: str | None = None
        self.current_run_dir: str | None = None

        # Objets H2O / AutoML
        self.aml: H2OAutoML | None = None
        self.test: h2o.H2OFrame | None = None
        self.best_model_path: str | None = None
        self.leaderboard_df: pd.DataFrame | None = None
        self.perf_test = None  # cache optionnel des performances sur test

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
            # si label est numérique mais correspond à une classification binaire, l'utilisateur veut factor
            # on ne force pas systématiquement : tenter de factoriser si cela fonctionne
            try:
                self.test["label"] = self.test["label"].asfactor()
            except Exception:
                pass
            return True
        except Exception as e:
            print(
                f"[ERREUR] Impossible de créer le H2OFrame de test ou de caster 'label' en factor : {e}"
            )
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
        time_budget: int = 60,
        *,
        run_name: str | None = None,
        export_checkpoints_subdir: str = "checkpoints",
        automl_kwargs: dict[str, Any] | None = None,
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

        # Convertir en DataFrame si c'est un numpy array (après StandardScaler)
        # IMPORTANT: Reset les index pour éviter les désalignements lors du concat
        X_train_df = (
            pd.DataFrame(self.X_train)
            if not isinstance(self.X_train, pd.DataFrame)
            else self.X_train.reset_index(drop=True)
        )

        # Gestion de X_test = None (cas où pas de jeu de test fourni)
        if self.X_test is not None:
            X_test_df = (
                pd.DataFrame(self.X_test)
                if not isinstance(self.X_test, pd.DataFrame)
                else self.X_test.reset_index(drop=True)
            )
        else:
            X_test_df = None

        # Extraire les valeurs pour ignorer l'index original (qui peut être désaligné après train_test_split)
        y_train_values = self.y_train.values if hasattr(self.y_train, "values") else self.y_train
        y_train_series = pd.Series(y_train_values, name="label")

        # DataFrames pandas → H2OFrame
        train_df = pd.concat([X_train_df, y_train_series], axis=1)
        train = h2o.H2OFrame(train_df)

        # Créer test H2OFrame seulement si X_test existe
        if X_test_df is not None:
            if self.y_test is not None:
                # Même traitement pour y_test: extraire les valeurs pour aligner l'index
                y_test_values = (
                    self.y_test.values if hasattr(self.y_test, "values") else self.y_test
                )
                y_test_series = pd.Series(y_test_values, name="label")
                test_df = pd.concat([X_test_df, y_test_series], axis=1)
            else:
                test_df = X_test_df
            self.test = h2o.H2OFrame(test_df)
        else:
            self.test = None
            print("[INFO] Pas de jeu de test fourni - évaluation sur test ignorée")

        y = "label"
        x = list(train.columns)
        if y in x:
            x.remove(y)

        # essayer de factoriser le label si possible
        try:
            train[y] = train[y].asfactor()
            if self.test is not None and y in self.test.columns:
                self.test[y] = self.test[y].asfactor()
        except Exception:
            pass

        # Construction des kwargs AutoML
        if automl_kwargs is None:
            automl_kwargs = {}

        export_checkpoints_dir = os.path.join(self.current_run_dir, export_checkpoints_subdir)
        self._ensure_dir(export_checkpoints_dir)

        print(f"[H2O] Lancement AutoML avec budget de {time_budget}s...", flush=True)
        print(f"[H2O] Features: {len(x)} colonnes", flush=True)
        print(f"[H2O] Training samples: {train.nrows}", flush=True)

        self.aml = H2OAutoML(
            max_runtime_secs=time_budget,
            project_name=run_name,
            seed=42,
            verbosity="info",
            export_checkpoints_dir=export_checkpoints_dir,
            **automl_kwargs,
        )
        self.aml.train(x=x, y=y, training_frame=train)

        # Afficher le leaderboard immédiatement après l'entraînement
        print("\n[H2O] === Leaderboard ===", flush=True)
        lb = get_leaderboard(self.aml, extra_columns="ALL")
        lb_df = lb.as_data_frame()
        print(lb_df.head(10).to_string(), flush=True)

        # Extraire le meilleur score du leaderboard (première ligne = meilleur modèle)
        # Les colonnes possibles : auc, logloss, mean_per_class_error, rmse, mse
        self.best_cv_score = None
        self.best_cv_metric = None
        for metric_col in ["auc", "mean_per_class_error", "logloss", "rmse", "mse"]:
            if metric_col in lb_df.columns:
                self.best_cv_score = lb_df[metric_col].iloc[0]
                self.best_cv_metric = metric_col
                print(
                    f"[H2O] Meilleur score CV ({metric_col}): {self.best_cv_score:.4f}", flush=True
                )
                break

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
            # Sauvegarder le modèle H2O (format h2o.save_model)
            self.best_model_path = h2o.save_model(
                model=best_model,
                path=dest_dir,
                force=True,
            )
            print(f"[SAVED] Meilleur modèle sauvegardé ici → {self.best_model_path}")
        except Exception as e:
            print(
                f"[AVERTISSEMENT] Impossible de sauvegarder le meilleur modèle (save_model) : {e}"
            )

        # Export MOJO si demandé (toujours dans le dossier du modèle leader)
        if self.save_mojo_flag:
            try:
                mojo_path = best_model.download_mojo(path=dest_dir, get_genmodel_jar=True)
                print(f"[SAVED] MOJO du meilleur modèle → {mojo_path}")
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de télécharger le MOJO du leader : {e}")

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
            # Utiliser pandas pour écrire le CSV (plus fiable cross-platform)
            self.leaderboard_df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"[SAVED] Leaderboard → {csv_path}")
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de sauvegarder la leaderboard en CSV : {e}")

        print(self.leaderboard_df.head(10))

    def list_runs(self) -> list[str]:
        """
        Liste les sous-dossiers de base_dir (chaque sous-dossier est un run possible).
        """
        if not os.path.isdir(self.base_dir):
            return []
        return sorted(
            d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))
        )

    # ------------------------------------------------------------------
    # Chargement d'un modèle existant
    # ------------------------------------------------------------------
    def charger_modele_existant(
        self,
        model_path: str | None = None,
        *,
        run_name: str | None = None,
        search_prefixes: list[str] | None = None,
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
        candidates = [f for f in fichiers if any(f.startswith(pref) for pref in search_prefixes)]

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
    def analyser_modele(self) -> dict[str, Any]:
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
            print(
                "[AVERTISSEMENT] current_run_dir est None, certains fichiers ne pourront pas être sauvegardés."
            )
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
                        print(
                            f"[AVERTISSEMENT] Impossible de sauvegarder l'importance des variables : {e}"
                        )
            else:
                print("[INFO] Aucune importance de variable disponible (vi est None).")
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de récupérer l'importance des variables : {e}")

        # 3) Plot varimp → PNG
        if self.save_varimp_plot_flag:
            print("\n[DEBUG] Génération du varimp_plot (sans blocage)...")
            try:
                # Utiliser la méthode built-in si disponible, sinon construire un plot pandas
                try:
                    best_model.varimp_plot(num_of_features=20)
                    plot_path = os.path.join(self.current_run_dir, "varimp_plot.png")
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                    resultats["varimp_plot_path"] = plot_path
                    print(f"[DEBUG] varimp_plot sauvegardé dans : {plot_path}")
                except Exception:
                    # fallback
                    if vi is not None and not vi.empty:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        vi_top = vi.head(20)
                        ax.barh(vi_top["variable"][::-1], vi_top["relative_importance"][::-1])
                        plt.tight_layout()
                        plot_path = os.path.join(self.current_run_dir, "varimp_plot.png")
                        plt.savefig(plot_path)
                        plt.close(fig)
                        resultats["varimp_plot_path"] = plot_path
                        print(f"[DEBUG] varimp_plot (fallback) sauvegardé dans : {plot_path}")
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de générer le varimp_plot : {e}")

        # 4) MOJO (déjà tenté dans save_best_model, mais on remet ici pour robustesse)
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

        # 5) Export complet JSON du modèle (raw)
        try:
            raw_json = best_model.as_json()
            raw_path = os.path.join(self.current_run_dir, f"{best_model.model_id}_model_raw.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_json, f, ensure_ascii=False, indent=2)
            print(f"[INFO] JSON brut du modèle sauvegardé → {raw_path}")
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de sauvegarder le JSON brut du leader : {e}")

        return resultats

    # ------------------------------------------------------------------
    # Exploration des features de tous les modèles
    # ------------------------------------------------------------------
    def sauvegarder_features_tous_modeles(self) -> dict[str, Any]:
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

        # créer dossier models/ pour stocker chaque modèle individuellement
        models_root = os.path.join(self.current_run_dir)
        self._ensure_dir(models_root)

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

                # --- Exporter les informations du modèle dans son dossier dédié ---
                model_dir = os.path.join(models_root, mid)
                self._ensure_dir(model_dir)

                # params.json
                try:
                    params = {k: v["actual"] for k, v in m.params.items()}
                    with open(os.path.join(model_dir, "params.json"), "w", encoding="utf-8") as f:
                        json.dump(params, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[WARN] Impossible de sauvegarder params.json pour {mid}: {e}")

                # summary.csv (essayer d'exporter un résumé si disponible)
                try:
                    summary = m.summary()
                    if summary is not None:
                        try:
                            summary_df = summary.as_data_frame()
                            summary_df.to_csv(
                                os.path.join(model_dir, "summary.csv"),
                                index=False,
                                encoding="utf-8",
                            )
                        except Exception:
                            # si summary n'est pas un frame pandas convertible, essayer str()
                            with open(
                                os.path.join(model_dir, "summary.txt"), "w", encoding="utf-8"
                            ) as f:
                                f.write(str(summary))
                except Exception as e:
                    print(f"[WARN] Impossible de générer summary pour {mid}: {e}")

                # variable_importance.csv
                try:
                    vi = m.varimp(use_pandas=True)
                    if vi is not None:
                        vi.to_csv(
                            os.path.join(model_dir, "variable_importance.csv"),
                            index=False,
                            encoding="utf-8",
                        )
                except Exception as e:
                    # Ne pas stopper si pas de varimp (ex: GLM sans varimp)
                    print(f"[DEBUG] Pas d'importance ou erreur pour varimp de {mid}: {e}")

                # model_raw.json
                try:
                    raw_json = m.as_json()
                    with open(
                        os.path.join(model_dir, "model_raw.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(raw_json, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[WARN] Impossible d'exporter model_raw.json pour {mid}: {e}")

                # MOJO
                try:
                    # Télécharger le mojo dans le dossier du modèle (certains modèles ne supportent pas MOJO)
                    m.download_mojo(path=model_dir, get_genmodel_jar=True)
                except Exception as e:
                    # échec non critique
                    print(f"[DEBUG] MOJO non disponible ou erreur pour {mid}: {e}")

                # Si stacked ensemble -> export base_models.csv + tenter d'exporter leurs MOJOs
                try:
                    if "StackedEnsemble" in mid:
                        base_models = getattr(m, "model_base_names", []) or []
                        df_base = pd.DataFrame(base_models, columns=["base_model_id"])
                        df_base.to_csv(
                            os.path.join(model_dir, "base_models.csv"),
                            index=False,
                            encoding="utf-8",
                        )

                        sub_dir = os.path.join(model_dir, "base_models")
                        self._ensure_dir(sub_dir)
                        for bm in base_models:
                            try:
                                sub_model = h2o.get_model(bm)
                                # tenter MOJO du sous modèle
                                try:
                                    sub_model.download_mojo(path=sub_dir, get_genmodel_jar=False)
                                except Exception:
                                    pass
                            except Exception as e:
                                print(f"[DEBUG] Impossible de charger sous-modèle {bm} : {e}")
                except Exception as e:
                    print(f"[DEBUG] Problème lors de l'export des sous-modèles pour {mid} : {e}")

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
                            lambda x: ",".join(map(str, x))
                            if isinstance(x, (list, tuple))
                            else str(x)
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

        - Affiche les performances H2O (si y_test disponible)
        - Affiche la matrice de confusion
        - Affiche Accuracy + F1-score
        - Sauvegarde un CSV avec X_test + prédictions
        """
        best_model = self._get_best_model()
        if best_model is None:
            return None

        if not self._ensure_test_frame():
            return None

        perf = None
        has_labels = "label" in self.test.columns and self.y_test is not None

        print(f"[INFO] Évaluation du modèle leader : {best_model.model_id}")

        # === Prédictions détaillées H2O ===
        try:
            preds = best_model.predict(self.test)
        except Exception as e:
            print(f"[ERREUR] Impossible de générer les prédictions : {e}")
            return None

        # === Si on a les labels, calculer les métriques ===
        if has_labels:
            # Performance brute H2O
            try:
                perf = best_model.model_performance(self.test)
                print("\n=== Performance sur le jeu de test ===")
                print(perf)
                self.perf_test = perf

                # Matrice de confusion
                try:
                    cm = perf.confusion_matrix()
                    if cm is not None:
                        print("\n=== Matrice de confusion (test) ===")
                        print(cm)
                except Exception as e:
                    print(f"[INFO] Pas de matrice de confusion disponible : {e}")
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de calculer la performance H2O : {e}")

            # Convertir en arrays numpy pour sklearn
            try:
                import numpy as np

                # Récupérer les prédictions
                y_pred_df = preds["predict"].as_data_frame()
                y_pred = y_pred_df.values.ravel()

                # Utiliser les y_test originaux si la taille correspond
                if self.y_test is not None:
                    y_test_values = (
                        self.y_test.values
                        if hasattr(self.y_test, "values")
                        else np.array(self.y_test)
                    )
                    if len(y_test_values) == len(y_pred):
                        # S'assurer que les types correspondent (convertir en int si nécessaire)
                        y_true = (
                            y_test_values.astype(int)
                            if np.issubdtype(y_test_values.dtype, np.number)
                            else y_test_values
                        )
                        # Convertir y_pred en int aussi pour la cohérence
                        y_pred = (
                            y_pred.astype(int) if np.issubdtype(y_pred.dtype, np.number) else y_pred
                        )
                        print("[INFO] Utilisation des y_test originaux pour les métriques")
                    else:
                        # Tailles différentes, utiliser les labels du H2OFrame
                        print(
                            f"[INFO] Taille y_test ({len(y_test_values)}) != y_pred ({len(y_pred)}), utilisation labels H2O"
                        )
                        y_true_df = self.test["label"].as_data_frame()
                        y_true = y_true_df.values.ravel()
                        # Convertir en int si possible
                        try:
                            y_true = y_true.astype(int)
                            y_pred = y_pred.astype(int)
                        except (ValueError, TypeError):
                            pass
                else:
                    # Fallback sur les labels H2O
                    y_true_df = self.test["label"].as_data_frame()
                    y_true = y_true_df.values.ravel()

                # Accuracy
                acc = accuracy_score(y_true, y_pred)
                print(f"\n=== Accuracy sur le test : {acc:.4f} ===")
                self.accuracy_test = acc

                # F1-scores
                f1_macro = f1_score(y_true, y_pred, average="macro")
                f1_micro = f1_score(y_true, y_pred, average="micro")
                f1_weighted = f1_score(y_true, y_pred, average="weighted")

                print("\n=== F1-score ===")
                print(f"F1 Macro     : {f1_macro:.4f}")
                print(f"F1 Micro     : {f1_micro:.4f}")
                print(f"F1 Weighted  : {f1_weighted:.4f}")

                self.f1_macro_test = f1_macro
                self.f1_micro_test = f1_micro
                self.f1_weighted_test = f1_weighted

            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de calculer les métriques sklearn : {e}")
        else:
            print("\n[INFO] Pas de labels dans le jeu de test (dataset Kaggle)")
            print("[INFO] Les métriques ne peuvent pas être calculées")
            print("[INFO] Seules les prédictions sont générées")

        # === Sauvegarde du CSV avec prédictions ===
        if self.save_predictions_flag and self.current_run_dir is not None:
            try:
                test_pd = self.test.as_data_frame()
                preds_pd = preds.as_data_frame()
                merged = pd.concat(
                    [test_pd.reset_index(drop=True), preds_pd.reset_index(drop=True)],
                    axis=1,
                )
                preds_path = os.path.join(self.current_run_dir, filename)
                merged.to_csv(preds_path, index=False, encoding="utf-8")
                print(f"\n[SAVED] Prédictions complètes → {preds_path}")
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible de sauvegarder le CSV : {e}")

        # Retourner le score AUC du training si pas de labels test
        if not has_labels and self.aml is not None:
            try:
                leader = self.aml.leader
                auc_train = leader.auc(train=True, xval=True)
                print(f"\n=== Score AUC (cross-validation) : {auc_train:.4f} ===")
                return auc_train
            except:
                pass

        return perf

    # ------------------------------------------------------------------
    # Pipeline complet: run + prédiction + arrêt cluster
    # ------------------------------------------------------------------
    def use_all(
        self,
        time_budget: int,
        *,
        run_name: str | None = None,
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
