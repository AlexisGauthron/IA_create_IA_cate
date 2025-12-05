"""Implementation of the llmfe pipeline."""

from __future__ import annotations

from collections.abc import Sequence

# from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from src.feature_engineering.llmfe import buffer, code_manipulation, evaluator, profile, sampler
from src.feature_engineering.llmfe import config as config_lib
from src.feature_engineering.llmfe.feature_formatter import FeatureFormat
from src.feature_engineering.llmfe.feature_insights import FeatureInsights

# Import conditionnel pour éviter les imports circulaires
if TYPE_CHECKING:
    from src.feature_engineering.path_config import FeatureEngineeringPathConfig


def _extract_function_names(specification: str) -> tuple[str, str]:
    """Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, "evaluate", "run"))
    if len(run_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@evaluate.run`.")
    evolve_functions = list(code_manipulation.yield_decorated(specification, "equation", "evolve"))

    if len(evolve_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@equation.evolve`.")

    return evolve_functions[0], run_functions[0]


def main(
    specification: str,
    inputs: Sequence[Any],
    config: config_lib.Config,
    meta_data: dict,
    max_sample_nums: Optional[int],
    class_config: config_lib.ClassConfig,
    path_config: Optional[FeatureEngineeringPathConfig] = None,
    feature_insights: Optional[FeatureInsights] = None,
    feature_format: FeatureFormat = FeatureFormat.BASIC,
    **kwargs,
):
    """Launch a llmfe experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        meta_data    : the metadata file containing the features.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
        path_config  : FeatureEngineeringPathConfig pour les chemins (optionnel).
        feature_insights: FeatureInsights pré-calculés (optionnel).
        feature_format: Format de présentation des features (BASIC, TAGS, HIERARCHICAL).
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    # Créer le buffer avec path_config et les nouveaux paramètres de formatage
    database = buffer.ExperienceBuffer(
        config.experience_buffer,
        template,
        function_to_evolve,
        meta_data,
        path_config=path_config,
        feature_insights=feature_insights,
        feature_format=feature_format,
    )

    # Créer le profiler avec path_config si disponible, sinon fallback sur log_dir
    log_dir = kwargs.get("log_dir", None)

    # Extraire les features originales et la colonne cible pour le tracker
    original_features = []
    target_column = None
    if isinstance(inputs, dict) and "data" in inputs:
        data_dict = inputs["data"]
        if "inputs" in data_dict and hasattr(data_dict["inputs"], "columns"):
            original_features = list(data_dict["inputs"].columns)
        # On peut aussi récupérer le nom de la target si disponible
        target_column = kwargs.get("target_column", None)

    if path_config is not None:
        profiler = profile.Profiler(
            path_config=path_config,
            original_features=original_features,
            target_column=target_column,
        )
        print(f"\n📁 Logs seront sauvegardés dans: {path_config.llmfe_dir}\n")
    elif log_dir is not None:
        profiler = profile.Profiler(
            log_dir=log_dir,
            original_features=original_features,
            target_column=target_column,
        )
    else:
        profiler = None

    # Passer le DataFrame original au profiler pour la sauvegarde du dataset transformé
    if profiler is not None and isinstance(inputs, dict) and "data" in inputs:
        data_dict = inputs["data"]
        if "inputs" in data_dict and "outputs" in data_dict and target_column:
            df_original = data_dict["inputs"].copy()
            df_original[target_column] = data_dict["outputs"]
            profiler.set_original_data(df_original, target_column)

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(
            evaluator.Evaluator(
                database,
                template,
                function_to_evolve,
                function_to_run,
                inputs,
                timeout_seconds=config.evaluate_timeout_seconds,
                sandbox_class=class_config.sandbox_class,
            )
        )
        print("Boucle")

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(
        initial,
        island_id=None,
        version_generated=None,
        data_input=inputs["data"]["inputs"],
        data_output=inputs["data"]["outputs"],
        profiler=profiler,
    )
    # Set global max sample nums.
    samplers = [
        sampler.Sampler(
            database,
            evaluators,
            config.samples_per_prompt,
            meta_data=meta_data,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
            config=config,
        )
        for _ in range(config.num_samplers)
    ]

    # Afficher le prompt initial avant la boucle
    print("\n" + "=" * 80)
    print("  PROMPT INITIAL ENVOYÉ AU LLM")
    print("=" * 80)
    initial_prompt = database.get_prompt()
    print(initial_prompt.code)
    print("=" * 80 + "\n")

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler)

    # Afficher le récapitulatif de tous les modèles à la fin
    if profiler is not None:
        profiler.print_summary(top_n=10)

    # Retourner les résultats pour la collecte de métriques
    results = {
        "scores": [],
        "n_features_per_iteration": [],
        "n_features_generated": 0,
        "best_score": 0.0,
        "final_score": 0.0,
        "n_iterations": 0,
    }

    if profiler is not None:
        results["scores"] = profiler.get_all_scores()
        results["n_features_per_iteration"] = profiler.get_feature_counts()
        results["n_features_generated"] = profiler.get_total_features_generated()
        results["best_score"] = profiler.get_best_score()
        results["final_score"] = results["scores"][-1] if results["scores"] else 0.0
        results["n_iterations"] = len(results["scores"])

    return results
