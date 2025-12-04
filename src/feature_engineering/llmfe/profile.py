# profile the experiment with tensorboard

from __future__ import annotations

import json
import logging
import os.path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from src.feature_engineering.llmfe import code_manipulation
from src.feature_engineering.llmfe.evolution_tracker import EvolutionTracker

# Import conditionnel pour éviter les imports circulaires
if TYPE_CHECKING:
    from src.feature_engineering.path_config import FeatureEngineeringPathConfig


class Profiler:
    def __init__(
        self,
        log_dir: str | None = None,
        pkl_dir: str | None = None,
        max_log_nums: int | None = None,
        path_config: FeatureEngineeringPathConfig | None = None,
        original_features: list[str] | None = None,
        target_column: str | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
            path_config : FeatureEngineeringPathConfig (prioritaire sur log_dir).
            original_features: Liste des features originales (pour EvolutionTracker).
            target_column: Colonne cible (pour EvolutionTracker).
        """
        logging.getLogger().setLevel(logging.INFO)
        self._path_config = path_config
        self._max_log_nums = max_log_nums

        # Utiliser path_config si fourni, sinon fallback sur log_dir
        # FeatureEngineeringPathConfig a: llmfe_tensorboard_dir, llmfe_samples_dir, llmfe_results_dir
        if path_config is not None:
            self._log_dir = str(path_config.llmfe_tensorboard_dir)
            self._json_dir = str(path_config.llmfe_samples_dir)
            self._results_dir = str(path_config.llmfe_results_dir)
        else:
            self._log_dir = log_dir
            self._json_dir = os.path.join(log_dir, "samples") if log_dir else None
            self._results_dir = None

        # Créer les dossiers nécessaires
        if self._json_dir:
            os.makedirs(self._json_dir, exist_ok=True)
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)

        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: dict[int, code_manipulation.Function] = {}

        if self._log_dir:
            self._writer = SummaryWriter(log_dir=self._log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

        # Initialiser l'EvolutionTracker
        tracker_output_dir = self._results_dir if self._results_dir else self._log_dir
        if tracker_output_dir:
            self._evolution_tracker = EvolutionTracker(
                output_dir=tracker_output_dir,
                original_features=original_features or [],
                target_column=target_column,
            )
        else:
            self._evolution_tracker = None

        # Stocker les infos pour la sauvegarde du dataset transformé
        self._target_column = target_column
        self._df_original = None  # Sera défini via set_original_data()

    def set_original_data(self, df: pd.DataFrame, target_col: str):
        """
        Stocke le DataFrame original pour la sauvegarde finale.

        Args:
            df: DataFrame original avec toutes les colonnes
            target_col: Nom de la colonne cible
        """
        self._df_original = df.copy()
        self._target_column = target_col

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            "Best Score of Function", self._cur_best_program_score, global_step=self._num_samples
        )
        self._writer.add_scalars(
            "Legal/Illegal Function",
            {
                "legal function num": self._evaluate_success_program_num,
                "illegal function num": self._evaluate_failed_program_num,
            },
            global_step=self._num_samples,
        )
        self._writer.add_scalars(
            "Total Sample/Evaluate Time",
            {"sample time": self._tot_sample_time, "evaluate time": self._tot_evaluate_time},
            global_step=self._num_samples,
        )

        # Log the function_str (seulement si non None)
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                "Best Function String", self._cur_best_program_str, global_step=self._num_samples
            )

    def _write_json(self, programs: code_manipulation.Function):
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score

        # Utiliser path_config si disponible pour un nom de fichier plus lisible
        # FeatureEngineeringPathConfig a get_llmfe_sample_path()
        if self._path_config is not None:
            path = self._path_config.get_llmfe_sample_path(sample_order)
            content = {
                "sample_order": sample_order,
                "function": function_str,
                "score": score,
                "timestamp": None,
            }
        else:
            path = os.path.join(self._json_dir, f"samples_{sample_order}.json")
            content = {"sample_order": sample_order, "function": function_str, "score": score}

        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(content, json_file, indent=2, ensure_ascii=False)

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)

            # Enregistrer dans l'EvolutionTracker
            if self._evolution_tracker is not None:
                function_str = str(programs).strip("\n")
                error_msg = None
                if programs.score is None:
                    error_msg = "Evaluation failed"

                self._evolution_tracker.record_sample(
                    sample_order=sample_orders if sample_orders is not None else 0,
                    score=programs.score,
                    function_code=function_str,
                    sample_time=programs.sample_time or 0.0,
                    evaluate_time=programs.evaluate_time or 0.0,
                    error=error_msg,
                )

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip("\n")
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print("================= Evaluated Function =================")
        print(f"{function_str}")
        print("------------------------------------------------------")
        print(f"Score        : {str(score)}")
        print(f"Sample time  : {str(sample_time)}")
        print(f"Evaluate time: {str(evaluate_time)}")
        print(f"Sample orders: {str(sample_orders)}")
        print("======================================================\n\n")

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def print_summary(self, top_n: int = 10):
        """
        Affiche un récapitulatif de tous les modèles générés.

        Args:
            top_n: Nombre de meilleurs modèles à afficher en détail (défaut: 10)
        """
        print("\n" + "=" * 70)
        print("                    RÉCAPITULATIF DES MODÈLES")
        print("=" * 70)

        # Statistiques générales
        print("\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"   • Total de modèles générés    : {self._num_samples}")
        print(f"   • Modèles valides (avec score): {self._evaluate_success_program_num}")
        print(f"   • Modèles échoués             : {self._evaluate_failed_program_num}")
        print(f"   • Temps total de sampling     : {self._tot_sample_time:.2f}s")
        print(f"   • Temps total d'évaluation    : {self._tot_evaluate_time:.2f}s")

        # Récupérer et trier tous les modèles par score
        all_functions = list(self._all_sampled_functions.values())
        valid_functions = [f for f in all_functions if f.score is not None]
        sorted_functions = sorted(valid_functions, key=lambda x: x.score, reverse=True)

        if not sorted_functions:
            print("\n⚠️  Aucun modèle valide n'a été généré.")
            print("=" * 70 + "\n")
            return

        # Meilleur modèle
        print("\n🏆 MEILLEUR MODÈLE:")
        print(f"   • Score: {self._cur_best_program_score}")
        print(f"   • Ordre d'échantillonnage: {self._cur_best_program_sample_order}")
        print("   • Code:")
        for line in self._cur_best_program_str.split("\n"):
            print(f"      {line}")

        # Top N modèles
        print(f"\n📈 TOP {min(top_n, len(sorted_functions))} MODÈLES (par score décroissant):")
        print("-" * 70)
        for i, func in enumerate(sorted_functions[:top_n], 1):
            print(f"\n   #{i} | Score: {func.score:.6f} | Sample order: {func.global_sample_nums}")
            func_str = str(func).strip()
            # Afficher les premières lignes du code
            lines = func_str.split("\n")
            for line in lines[:8]:  # Limiter à 8 lignes par fonction
                print(f"      {line}")
            if len(lines) > 8:
                print(f"      ... ({len(lines) - 8} lignes supplémentaires)")
            print("-" * 70)

        # Distribution des scores
        if len(sorted_functions) > 1:
            scores = [f.score for f in sorted_functions]
            print("\n📉 DISTRIBUTION DES SCORES:")
            print(f"   • Score max    : {max(scores):.6f}")
            print(f"   • Score min    : {min(scores):.6f}")
            print(f"   • Score moyen  : {sum(scores)/len(scores):.6f}")
            print(
                f"   • Écart-type   : {(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.6f}"
            )

        # Chemin des logs
        if self._log_dir:
            print("\n📁 FICHIERS DE LOGS:")
            print(f"   • TensorBoard : {self._log_dir}")
            print(f"   • JSON samples: {self._json_dir}")

        print("\n" + "=" * 70)
        print("              FIN DU RÉCAPITULATIF")
        print("=" * 70 + "\n")

        # Sauvegarder les résultats si path_config est disponible
        if self._path_config is not None:
            self._save_final_results(sorted_functions)

        # Générer le rapport d'évolution avec l'EvolutionTracker
        if self._evolution_tracker is not None:
            self._evolution_tracker.print_evolution_table()
            self._evolution_tracker.save()
            self._evolution_tracker.save_parquet()
            self._evolution_tracker.generate_report()

    def _save_final_results(self, sorted_functions: list[code_manipulation.Function]):
        """Sauvegarde les résultats finaux dans le dossier results."""
        if self._path_config is None:
            return

        # Sauvegarder le meilleur modèle
        if self._cur_best_program_str:
            best_model_info = {
                "score": self._cur_best_program_score,
                "sample_order": self._cur_best_program_sample_order,
                "function_code": self._cur_best_program_str,
            }
            self._path_config.save_llmfe_best_model(best_model_info)

        # Sauvegarder tous les scores
        all_scores = [
            {
                "sample_order": func.global_sample_nums,
                "score": func.score,
            }
            for func in sorted_functions
        ]
        self._path_config.save_llmfe_scores(all_scores)

        # Sauvegarder le résumé dans le dossier results
        summary = {
            "total_samples": self._num_samples,
            "valid_samples": self._evaluate_success_program_num,
            "failed_samples": self._evaluate_failed_program_num,
            "total_sample_time": self._tot_sample_time,
            "total_evaluate_time": self._tot_evaluate_time,
            "best_score": self._cur_best_program_score,
            "best_sample_order": self._cur_best_program_sample_order,
        }

        if sorted_functions:
            scores = [f.score for f in sorted_functions]
            summary["score_stats"] = {
                "max": max(scores),
                "min": min(scores),
                "mean": sum(scores) / len(scores),
            }

        # Sauvegarder le summary dans llmfe_results_dir
        summary_path = self._path_config.llmfe_results_dir / "summary.json"
        self._path_config.save_json(summary, summary_path)
        print(f"\n✅ Résultats sauvegardés dans: {self._path_config.llmfe_results_dir}")

        # Appliquer modify_features et sauvegarder le dataset transformé
        self._save_transformed_dataset()

    def _save_transformed_dataset(self):
        """Applique la meilleure fonction modify_features et sauvegarde le dataset."""
        if not self._cur_best_program_str:
            print("⚠️ Pas de meilleur programme trouvé - dataset non transformé")
            return

        if self._df_original is None:
            print("⚠️ DataFrame original non défini - dataset non sauvegardé")
            return

        if self._target_column is None:
            print("⚠️ Colonne cible non définie - dataset non sauvegardé")
            return

        try:
            # Exécuter le code pour obtenir la fonction modify_features
            local_vars = {}
            exec(self._cur_best_program_str, {"pd": pd, "np": np}, local_vars)
            modify_features = local_vars.get("modify_features")

            if modify_features is None:
                print("⚠️ Fonction modify_features non trouvée dans le code")
                return

            # Appliquer la transformation sur les features (sans la target)
            X = self._df_original.drop(columns=[self._target_column])
            df_transformed = modify_features(X)

            # Réajouter la colonne cible
            df_transformed[self._target_column] = self._df_original[self._target_column].values

            # Sauvegarder en CSV
            self._path_config.save_transformed_dataset(df_transformed)
            print(f"✅ Dataset transformé sauvegardé dans: {self._path_config.dataset_fe_dir}")
            print(f"   - {len(df_transformed)} lignes, {len(df_transformed.columns)} colonnes")
            print(
                f"   - Nouvelles features: {len(df_transformed.columns) - len(self._df_original.columns)}"
            )

        except Exception as e:
            print(f"⚠️ Erreur lors de l'application de modify_features: {e}")
            import traceback

            traceback.print_exc()
