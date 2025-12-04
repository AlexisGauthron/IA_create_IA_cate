"""A multi-island experience buffer that implements the evolutionary algorithm."""

from __future__ import annotations

import copy
import dataclasses
import os
import profile
import random
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy
from absl import logging

from src.feature_engineering.llmfe import code_manipulation
from src.feature_engineering.llmfe import config as config_lib
from src.feature_engineering.llmfe.feature_formatter import FeatureFormat, FeatureFormatter
from src.feature_engineering.llmfe.feature_insights import FeatureInsights

# Import conditionnel pour éviter les imports circulaires
if TYPE_CHECKING:
    from src.feature_engineering.path_config import FeatureEngineeringPathConfig

from src.feature_engineering.llmfe.utils import is_categorical, serialize

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the Experience Buffer, to be sent to Samplers.

    Args:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the samples
                included in the prompt. Used to direct the newly generated sample
                into the same island.
    """

    code: str
    version_generated: int
    island_id: int
    data_input: pd.DataFrame
    data_output: list


class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        config: config_lib.ExperienceBufferConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
        meta_data: dict,
        path_config: FeatureEngineeringPathConfig | None = None,
        feature_insights: FeatureInsights | None = None,
        feature_format: FeatureFormat = FeatureFormat.BASIC,
    ) -> None:
        self._config: config_lib.ExperienceBufferConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._meta_data: dict = meta_data
        self._path_config: FeatureEngineeringPathConfig | None = path_config
        self._feature_insights: FeatureInsights | None = feature_insights
        self._feature_format: FeatureFormat = feature_format

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(
                    template,
                    function_to_evolve,
                    config.functions_per_prompt,
                    meta_data,
                    config.cluster_sampling_temperature_init,
                    config.cluster_sampling_temperature_period,
                    path_config=path_config,
                    feature_insights=feature_insights,
                    feature_format=feature_format,
                )
            )
        self._best_score_per_island: list[float] = [-float("inf")] * config.num_islands
        self._best_program_per_island: list[code_manipulation.Function | None] = [
            None
        ] * config.num_islands
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = [
            None
        ] * config.num_islands

        self._last_reset_time: float = time.time()

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing samples from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated, data_input, data_output = self._islands[island_id].get_prompt()

        return Prompt(code, version_generated, island_id, data_input, data_output)

    def _register_program_in_island(
        self,
        program: code_manipulation.Function,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        island_id: int,
        scores_per_test: ScoresPerTest,
        **kwargs,
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(input_data, output_data, program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info("Best score of island %d increased to %s", island_id, score)

        profiler: profile.Profiler = kwargs.get("profiler", None)
        if profiler:
            global_sample_nums = kwargs.get("global_sample_nums", None)
            sample_time = kwargs.get("sample_time", None)
            evaluate_time = kwargs.get("evaluate_time", None)
            program.data_input = input_data
            program.data_output = output_data
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)

    def register_program(
        self,
        program: code_manipulation.Function,
        island_id: int | None,
        scores_per_test: ScoresPerTest,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        **kwargs,
    ) -> None:
        """Registers new `program` skeleton hypotheses in the experience buffer."""
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(
                    program, input_data, output_data, island_id, scores_per_test, **kwargs
                )
        else:
            self._register_program_in_island(
                program, input_data, output_data, island_id, scores_per_test, **kwargs
            )

        # Check island reset
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # Sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island + np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._meta_data,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
                path_config=self._path_config,
                feature_insights=self._feature_insights,
                feature_format=self._feature_format,
            )
            self._best_score_per_island[island_id] = -float("inf")
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, None, None, island_id, founder_scores)


class Island:
    """A sub-population of the program skeleton experience buffer."""

    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        functions_per_prompt: int,
        meta_data: dict,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
        path_config: FeatureEngineeringPathConfig | None = None,
        feature_insights: FeatureInsights | None = None,
        feature_format: FeatureFormat = FeatureFormat.BASIC,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._meta_data: dict = meta_data
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self._path_config: FeatureEngineeringPathConfig | None = path_config
        self._feature_insights: FeatureInsights | None = feature_insights
        self._feature_format: FeatureFormat = feature_format

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
        self,
        data_input: pd.DataFrame,
        data_output: list,
        program: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program, data_input, data_output)
        else:
            self._clusters[signature].register_program(program, data_input, data_output)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array([self._clusters[signature].score for signature in signatures])

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
            1 - (self._num_programs % period) / period
        )
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return (
            self._generate_prompt(sorted_implementations),
            version_generated,
            sorted_implementations[-1].data_input,
            sorted_implementations[-1].data_output,
        )

    def _generate_prompt(self, implementations: Sequence[code_manipulation.Function]) -> str:
        """Create a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)
        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        input_data = []
        output_data = []
        for i, implementation in enumerate(implementations):
            new_function_name = f"{self._function_to_evolve}_v{i}"
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f"Improved version of `{self._function_to_evolve}_v{i - 1}`."
                )
            # If the function is recursive, replace calls to itself with its new name.
            input_data.append(implementation.data_input)
            output_data.append(implementation.data_output)
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name
            )

            versioned_functions.append(code_manipulation.text_to_function(implementation))

        # Create header of new function to be completed
        next_version = len(implementations)
        new_function_name = f"{self._function_to_evolve}_v{next_version}"
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body="",
            docstring=(
                "Improved version of "
                f"`{self._function_to_evolve}_v{next_version - 1}`. Think and suggest new features."
            ),
        )
        versioned_functions.append(header)
        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)

        in_context_desc = ""
        df_input = input_data[-1].copy()
        if not df_input.index.name:
            df_input.reset_index(inplace=True, drop=True)
        if any(col.startswith("Unnamed:") for col in df_input.columns):
            # Drop the 'Unnamed:' column(s)
            df_input = df_input.loc[:, ~df_input.columns.str.startswith("Unnamed:")]

        df_output = pd.DataFrame(output_data[-1], columns=["Result"])
        df_current = df_input.join(df_output)
        df_current = df_current.sample(frac=1).head(10)

        total_column_list = [df_current.columns.tolist()]
        categorical_indicator = [
            is_categorical(df_current.iloc[:, i]) for i in range(df_current.shape[1])
        ]

        for selected_column in total_column_list:
            for icl_idx, icl_row in df_current.iterrows():
                icl_row = icl_row[selected_column]
                in_context_desc += serialize(icl_row)
                in_context_desc += "\n"

            # Utiliser le nouveau système de formatage si disponible
            if self._feature_insights is not None:
                # Utiliser FeatureFormatter avec les insights pré-calculés
                formatter = FeatureFormatter(
                    insights=self._feature_insights,
                    format_type=self._feature_format,
                )
                feature_desc = formatter.format()
            else:
                # Fallback : format basique calculé à la volée
                feature_desc = self._generate_feature_desc_legacy(
                    selected_column, df_current, input_data[-1], categorical_indicator
                )

        new_prompt = str(prompt)
        random_number = random.randint(0, 2)

        # Déterminer le type de prompt (operations ou domain)
        prompt_type = "operations" if random_number == 0 else "domain"

        # Charger les prompts via path_config si disponible, sinon fallback
        if self._path_config is not None:
            try:
                prefix = self._path_config.read_prompt(prompt_type, "head")
                suffix = self._path_config.read_prompt(prompt_type, "tail")
            except FileNotFoundError:
                logging.warning(
                    f"Prompts {prompt_type} non trouvés via path_config, utilisation du fallback"
                )
                prefix, suffix = self._load_prompts_fallback(prompt_type)
        else:
            prefix, suffix = self._load_prompts_fallback(prompt_type)

        new_prompt = new_prompt.replace("[PREFIX]", prefix).replace("[SUFFIX]", suffix)
        new_prompt = new_prompt.replace("[EXAMPLES]", in_context_desc).replace(
            "[FEATURES]", feature_desc
        )

        return new_prompt

    def _load_prompts_fallback(self, prompt_type: str) -> tuple[str, str]:
        """Charge les prompts avec l'ancienne méthode (chemins relatifs)."""
        prefix = ""
        suffix = ""

        template_prefix = os.path.join("prompts", f"{prompt_type}_head.txt")
        if os.path.exists(template_prefix):
            with open(template_prefix) as f:
                prefix = f.read()

        template_suffix = os.path.join("prompts", f"{prompt_type}_tail.txt")
        if os.path.exists(template_suffix):
            with open(template_suffix) as f:
                suffix = f.read()

        return prefix, suffix

    def _generate_feature_desc_legacy(
        self,
        selected_column: list,
        df_current: pd.DataFrame,
        input_data: pd.DataFrame,
        categorical_indicator: list,
    ) -> str:
        """
        Génère la description des features avec l'ancienne méthode (format BASIC).
        Utilisé comme fallback quand feature_insights n'est pas fourni.
        """
        feature_name_list = []
        sel_cat_idx = [df_current.columns.tolist().index(col_name) for col_name in selected_column]
        is_cat_sel = np.array(categorical_indicator)[sel_cat_idx]

        for cidx, cname in enumerate(selected_column):
            if cname == "Result":
                break
            if is_cat_sel[cidx] == True:
                clist = df_current[cname].unique().tolist()
                clist = [str(c) for c in clist]

                clist_str = ", ".join(clist)
                desc = self._meta_data[cname] if cname in self._meta_data.keys() else ""
                feature_name_list.append(
                    f"- {cname}: {desc} (categorical variable with categories [{clist_str}])"
                )
            else:
                # Protection contre les types mixtes (float + str)
                try:
                    col_data = input_data[cname]
                    # Filtrer seulement les valeurs numériques
                    numeric_data = pd.to_numeric(col_data, errors="coerce")
                    min_val = numeric_data.min()
                    max_val = numeric_data.max()
                    if pd.isna(min_val) or pd.isna(max_val):
                        # Si pas de valeurs numériques, traiter comme catégoriel
                        clist = col_data.dropna().unique().tolist()[:5]
                        clist_str = ", ".join([str(c) for c in clist])
                        desc = self._meta_data.get(cname, cname.replace("_", " "))
                        feature_name_list.append(
                            f"- {cname}: {desc} (mixed type with values [{clist_str}])"
                        )
                    else:
                        desc = self._meta_data.get(cname, cname.replace("_", " "))
                        feature_name_list.append(
                            f"- {cname}: {desc} (numerical variable within range [{min_val}, {max_val}])"
                        )
                except Exception:
                    # Fallback en cas d'erreur
                    desc = self._meta_data.get(cname, cname.replace("_", " "))
                    feature_name_list.append(f"- {cname}: {desc} (feature)")

        return "\n".join(feature_name_list)


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(
        self,
        score: float,
        implementation: code_manipulation.Function,
        data_input: pd.DataFrame,
        data_output: pd.DataFrame,
    ):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]
        self._data_input: list[pd.DataFrame] = [data_input]
        self._data_output: list[array] = [data_output]

    @property
    def score(self) -> float:
        return self._score

    def register_program(
        self,
        program: code_manipulation.Function,
        data_input: pd.DataFrame,
        data_output: pd.DataFrame,
    ) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))
        self._data_input.append(data_input)
        self._data_output.append(data_output)

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
            max(self._lengths) + 1e-6
        )
        probabilities = _softmax(-normalized_lengths, temperature=1.0)

        return np.random.choice(self._programs, p=probabilities)
