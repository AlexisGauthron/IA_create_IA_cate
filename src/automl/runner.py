import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import os, traceback, logging
from datetime import datetime
from typing import Tuple, Optional

# --- utilitaires communs ------------------------------------------------------
def _ensure_logger(self):
    print("\n")
    print("#"*70)
    print(f"[ERROR] Regarder dans les logs de Modeles")
    print("#"*70)
    print("\n")
    if getattr(self, "_logger", None):
        return self._logger
    log_dir = os.path.join(self.Nom_dossier, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"automl_{datetime.now():%Y%m%d}.log")

    logger = logging.getLogger(f"automl_{id(self)}")
    logger.setLevel(logging.INFO)
    # éviter les handlers en double
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    self._logger = logger
    self._log_path = log_path
    if not hasattr(self, "errors"):
        self.errors = {}
    return logger

def _record_error(self, name: str, exc: Exception, hint: Optional[str] = None):
    logger = _ensure_logger(self)
    tb = traceback.format_exc()
    logger.error("[%s] %s: %s\n%s", name, exc.__class__.__name__, str(exc), tb)
    self.errors[name] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "hint": hint,
        "traceback": tb,
    }





class all_autoML:

    def __init__(self,Nom_dossier : str, X_train, X_test, y_train, y_test = None):
        self.Nom_dossier = Nom_dossier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.score_flaml = None
        self.score_autogluon = None
        self.score_tpot = None
        self.score_h2o = None




    # --- implémentations avec gestion d'erreurs -----------------------------------
    def flaml(self):
        logger = _ensure_logger(self)
        self.score_flaml = None
        try:
            try:
                import src.automl.supervised.flaml_wrapper as auto_flaml
            except ImportError as e:
                _record_error(self, "flaml.import", e, "Vérifie l'installation: `poetry add flaml[automl]`.")
                return None

            Flaml = auto_flaml.autoMl_flaml(self.Nom_dossier, self.X_train, self.X_test, self.y_train, self.y_test)
            Flaml.flaml()  # entraînement
            self.score_flaml = Flaml.predict_test()  # évaluation

            # Enregistrer le modèle (best effort)
            try:
                Flaml.enregistrement_model()
            except Exception as e:
                _record_error(self, "flaml.save", e, "Vérifie les droits/chemins d’export.")
        except MemoryError as e:
            _record_error(self, "flaml", e, "Réduis time_budget, n_jobs, taille des données ou libère de la RAM.")
        except ValueError as e:
            _record_error(self, "flaml", e, "Contrôle les NaN/Inf, types de colonnes et la cohérence de la cible.")
        except Exception as e:
            _record_error(self, "flaml", e, "Consulte le log pour le détail.")
        return self.score_flaml


    def autogluon(self):
        logger = _ensure_logger(self)
        self.score_autogluon = None
        try:
            try:
                import src.automl.supervised.autogluon_wrapper as auto_gluon
            except ImportError as e:
                _record_error(self, "autogluon.import", e, "Installe: `poetry add autogluon`.")
                return None

            Autogluon = auto_gluon.autoMl_autogluon(self.Nom_dossier, self.X_train, self.X_test, self.y_train, self.y_test)
            Autogluon.autogluon()               # entraînement
            self.score_autogluon = Autogluon.predict_test()

        except ModuleNotFoundError as e:
            # cas fréquents: xgboost/lightgbm non installés
            _record_error(self, "autogluon", e, "Installe les backends manquants (ex: `poetry add lightgbm xgboost`).")
        except MemoryError as e:
            _record_error(self, "autogluon", e, "Active subsampling, réduis hyperparameter_tune/num_stack_levels.")
        except ValueError as e:
            _record_error(self, "autogluon", e, "Vérifie les types, NaN, classe cible, et le problème (reg/classif).")
        except Exception as e:
            _record_error(self, "autogluon", e, "Consulte le log pour le détail.")
        return self.score_autogluon


    def tpot(self):
        logger = _ensure_logger(self)
        self.score_tpot = None
        try:
            try:
                import src.automl.supervised.tpot_wrapper as auto_tpot
            except ImportError as e:
                _record_error(self, "tpot.import", e, "Installe: `poetry add tpot scikit-learn`.")
                return None

            Tpot = auto_tpot.autoMl_tpot(self.Nom_dossier, self.X_train, self.X_test, self.y_train, self.y_test)
            Tpot.tpot1()                        # entraînement
            self.score_tpot = Tpot.predict_test()

            try:
                Tpot.enregistrement_model()     # export pipeline
            except Exception as e:
                _record_error(self, "tpot.save", e, "Vérifie le chemin d’export et les permissions.")
        except ValueError as e:
            hint = None
            msg = str(e).lower()
            if "median strategy" in msg and "non-numeric" in msg:
                hint = ("Colonnes non numériques détectées. "
                        "Encode les catégorielles (One-Hot) ou passe imputation_strategy='most_frequent'.")
            _record_error(self, "tpot", e, hint or "Vérifie les dtypes, NaN et la config TPOT.")
        except MemoryError as e:
            _record_error(self, "tpot", e, "Réduis generations/population_size, n_jobs, ou échantillonne.")
        except Exception as e:
            _record_error(self, "tpot", e, "Consulte le log pour le détail.")
        return self.score_tpot


    def h2o(self,time_budget):
        logger = _ensure_logger(self)
        self.score_h2o = None
        h2o = None
        try:
            try:
                import src.automl.supervised.h2o_wrapper as auto_h2o
            except ImportError as e:
                _record_error(self, "h2o.import", e, "Vérifie le module wrapper src.automl.supervised.h2o_wrapper.")
                return None

            try:
                import h2o  # pour fermer proprement en finally si besoin
                import h2o.exceptions as h2o_ex
            except ImportError as e:
                _record_error(self, "h2o.runtime", e, "Installe: `poetry add h2o` et assure-toi d’avoir Java (JRE/JDK).")
                return None

            H2o = auto_h2o.autoMl_h2o(self.Nom_dossier, self.X_train, self.X_test, self.y_train, self.y_test,
                                      save_best_model=True,
                                      save_features_csv=True,
                                      save_features_json=False,
                                      save_leaderboard=True,
                                      save_mojo=False,
                                      save_predictions=True,
                                      save_varimp_csv=True,
                                      save_varimp_plot=False)

            print(f"[INFO] Démarrage H2O AutoML (budget: {time_budget}s)...", flush=True)
            H2o.use_all(time_budget=time_budget)
            # Récupérer le F1 score macro, ou accuracy si F1 non disponible
            self.score_h2o = getattr(H2o, 'f1_macro_test', None)
            if self.score_h2o is None:
                self.score_h2o = getattr(H2o, 'accuracy_test', None)
            # Si toujours None, essayer d'obtenir le AUC depuis perf_test
            if self.score_h2o is None and getattr(H2o, 'perf_test', None) is not None:
                try:
                    self.score_h2o = H2o.perf_test.auc()
                except Exception:
                    pass
            # Fallback: utiliser le meilleur score du leaderboard (validation croisée)
            if self.score_h2o is None:
                self.score_h2o = getattr(H2o, 'best_cv_score', None)
                if self.score_h2o is not None:
                    metric_name = getattr(H2o, 'best_cv_metric', 'cv')
                    print(f"[INFO] Score H2O (CV {metric_name}): {self.score_h2o:.4f}", flush=True)
            print(f"[INFO] FIN H2O AutoML (score: {self.score_h2o})", flush=True)

        except Exception as e:
            # Détection de messages fréquents pour fournir un hint utile
            msg = str(e).lower()
            hint = None
            if "illegal in h2oautoml id" in msg or "character '/' is illegal" in msg:
                hint = "Retire les '/' de project_name. Utilise '_' ou un identifiant simple."
            elif "java" in msg and ("not found" in msg or "failed" in msg):
                hint = "Java requis. Installe un JDK 8+ (ex: Temurin) et assure JAVA_HOME/PATH."
            elif "connection" in msg or "port 54321" in msg:
                hint = "Port occupé ou cluster existant. Ferme l’ancien cluster ou change le port."
            _record_error(self, "h2o", e, hint or "Vérifie Java, project_name, RAM et les permissions disque.")

        finally:
            print("[DEBUG] FINISH\n\n")
            # On essaie de fermer proprement le cluster si on l'a importé
            try:
                if h2o is not None and h2o.connection() is not None:
                    # ne pas planter si déjà fermé
                    pass
            except Exception:
                pass
        return self.score_h2o


    def h2o_all(self,time_zone=[100]):
        score_h2o = []
        for i in time_zone:
            score_h2o.append(self.h2o(i))
        
        max_score_h2o = max(score_h2o)

        return max_score_h2o



    def compare_all_predict(self,model):
        print("\n[INFO] all\n\n")
        print("#####################################################\n\n")
        if "flaml" in model:
            print(f"score_flaml : {self.score_flaml}\n")
        if "autogluon" in model:
            print(f"score_autogluon : {self.score_autogluon}\n")
        if "tpot" in model:
            print(f"score_tpot : {self.score_tpot}\n")
        if "h2o" in model:
            print(f"score_h2o : {self.score_h2o}\n")
        print("#####################################################\n\n")



    def use_all(self,model = ["autogluon","flaml","tpot","h2o"]):
        if "autogluon" in model:
            self.autogluon()
        if "flaml" in model:
            self.flaml()
        if "tpot" in model:
            self.tpot()
        if "h2o" in model:
            self.h2o_all()

        self.compare_all_predict(model)


# =============================================================================
# Fonction d'interface pour Streamlit
# =============================================================================

def run_automl(
    df_train,
    target_col: str,
    framework: str = "autogluon",
    time_limit: int = 600,
    project_name: str = "default",
) -> dict:
    """
    Lance AutoML sur le dataset fourni.

    Args:
        df_train: DataFrame d'entraînement avec la cible
        target_col: Nom de la colonne cible
        framework: Framework à utiliser (autogluon, flaml, h2o, tpot)
        time_limit: Temps limite en secondes
        project_name: Nom du projet pour les outputs

    Returns:
        Dictionnaire avec les résultats
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Séparer X et y
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    # Split train/test pour évaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Créer le dossier de sortie
    output_dir = os.path.join("outputs", project_name, "automl")
    os.makedirs(output_dir, exist_ok=True)

    # Initialiser AutoML
    automl = all_autoML(
        Nom_dossier=output_dir,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # Lancer le framework sélectionné
    score = None
    best_model = framework

    if framework == "autogluon":
        score = automl.autogluon()
    elif framework == "flaml":
        score = automl.flaml()
    elif framework == "h2o":
        score = automl.h2o(time_budget=time_limit)
    elif framework == "tpot":
        score = automl.tpot()
    else:
        raise ValueError(f"Framework inconnu: {framework}")

    # Compiler les résultats
    results = {
        "best_model": best_model,
        "best_score": score,
        "framework": framework,
        "time_limit": time_limit,
        "output_dir": output_dir,
        "errors": getattr(automl, "errors", {}),
    }

    return results