"""
Wrapper pour FLAML AutoML.
"""

from __future__ import annotations

import joblib
from flaml import AutoML
from sklearn.metrics import f1_score

# =============================================================================
# Classe FlamlWrapper
# =============================================================================
# Renommée de 'autoMl_flaml' vers 'FlamlWrapper'
# Raison: 'autoMl_flaml' mélangeait camelCase et snake_case
#         Les classes doivent être en PascalCase
# =============================================================================


class FlamlWrapper:
    """
    Wrapper pour le framework FLAML AutoML.

    Exemple:
        wrapper = FlamlWrapper(
            output_dir="outputs/projet",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        wrapper.flaml(time_budget=120)
        score = wrapper.predict_test()
    """

    def __init__(
        self,
        output_dir: str,  # Renommé de 'Nom_dossier' → snake_case anglais
        X_train,
        X_test,
        y_train,
        y_test,
    ):
        """
        Initialise le wrapper FLAML.

        Args:
            output_dir: Dossier de sortie pour les logs et modèles
                        (anciennement 'Nom_dossier')
            X_train: Features d'entraînement
            X_test: Features de test
            y_train: Cible d'entraînement
            y_test: Cible de test
        """
        self.output_dir = output_dir  # Renommé de 'Nom_dossier'
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.automl = None

    def flaml(
        self,
        task: str = "classification",
        metric: str = "auto",
        time_budget: int = 60,
        taille_max_modele: int = None,
        espace_ram_libre: int = 0,
    ):
        """
        Lance l'entraînement FLAML.

        Args:
            task: Type de tâche ("classification" ou "regression")
            metric: Métrique d'optimisation ("auto" pour détection automatique)
            time_budget: Budget temps en secondes
            taille_max_modele: Taille max d'un modèle en mémoire
            espace_ram_libre: Ratio de RAM à garder libre
        """
        print("[INFO] Recherche Meilleur Modele flaml\n")

        # Détection automatique de la métrique selon le type de problème
        if metric == "auto":
            y_values = self.y_train.values if hasattr(self.y_train, "values") else self.y_train
            n_classes = len(set(y_values))
            if n_classes == 2:
                metric = "f1"
                print("[INFO] Problème binaire détecté -> métrique: f1")
            else:
                metric = "macro_f1"  # FLAML utilise "macro_f1" et non "f1_macro"
                print(
                    f"[INFO] Problème multiclasse détecté ({n_classes} classes) -> métrique: macro_f1"
                )

        self.automl = AutoML()
        self.automl.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            task=task,
            metric=metric,
            time_budget=time_budget,
            log_file_name=f"{self.output_dir}/flaml.log",
            verbose=3,
            mem_thres=taille_max_modele,
            free_mem_ratio=espace_ram_libre,
        )

        print("Meilleur algo :", self.automl.best_estimator)
        print("Meilleure config :", self.automl.best_config)
        print("Perte (proxy) :", self.automl.best_loss)

    def predict_test(self):
        """Évalue le modèle sur le jeu de test."""
        print("[INFO] Test\n")

        # Si X_test ou y_test est None, retourner le score de validation
        if self.X_test is None or self.y_test is None:
            print("[INFO] Pas de jeu de test disponible ou pas de labels")
            print("[INFO] Retour du score de validation (1 - best_loss)")
            # FLAML stocke best_loss qui est 1-metric pour les métriques de score
            if self.automl.best_loss is not None:
                score = 1 - self.automl.best_loss
                print(f"[INFO] Score de validation: {score:.4f}\n")
                return score
            return None

        pred = self.automl.predict(self.X_test)

        # Adapter le calcul F1 selon le nombre de classes
        y_test_values = self.y_test.values if hasattr(self.y_test, "values") else self.y_test
        n_classes = len(set(y_test_values))
        if n_classes == 2:
            score = f1_score(self.y_test, pred)
        else:
            score = f1_score(self.y_test, pred, average="macro")
        print(f"F1{'_macro' if n_classes > 2 else ''}: {score}\n")
        return score

    def enregistrement_model(self):
        """Sauvegarde le modèle entraîné."""
        print("[INFO] Enregistrement model\n")
        joblib.dump(self.automl.model, f"{self.output_dir}/flaml_model.joblib")

    def chargement_model(self):
        """Charge un modèle sauvegardé."""
        print("[INFO] Chargement model\n")
        model = joblib.load(f"{self.output_dir}/flaml_model.joblib")
        return model
