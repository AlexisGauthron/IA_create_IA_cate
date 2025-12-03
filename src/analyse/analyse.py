from __future__ import annotations

from typing import Union, Sequence
import pandas as pd


import src.analyse.statistiques.report as report
from src.analyse.helper.compress_data import compact_llm_snapshot_payload
from src.analyse.metier.chatbot_llm import OllamaClient
from src.analyse.metier.chatbot_llm import BusinessClarificationBot
from src.analyse.metier.parsing_json import apply_llm_business_annotations
import src.analyse.statistiques.write_json as write_json


def analyse(df: pd.DataFrame,
            target_cols: Union[str, Sequence[str]],
            nom : str,
            print_json : bool = False,
            chemin_json : str = "json",
            provider: str = "ollama",    # openai
            model_metier: str = "deepseek-r1:8b",  #gpt-4.1-mini
            only_stats: bool = False,
            # Options pour les corrélations
            with_correlations: bool = False,
            correlation_methods: list = None,
            correlation_task: str = "classification",
    ):

    reports = report.analyze_dataset_for_fe(
        df,
        target_cols=target_cols,
        print_report=True,
        with_correlations=with_correlations,
        correlation_methods=correlation_methods,
        correlation_task=correlation_task,
    )

    reports_stat = reports["llm_payload"]

    if print_json:
        write_json.save_report_to_json(
            report=reports_stat,
            output_path=f"Test/analyse/{chemin_json}/stats/test_analyse_metier_report_{nom}.json",
        )

    if not only_stats:
        compact_payload = compact_llm_snapshot_payload(
            payload=reports_stat,
            max_example_values=3,
            max_top_values=3,
            float_ndigits=4,
            feature_engineering = False
        )

        print("=== LLM Feature Engineering Pipeline ===\n")
        
        llm = OllamaClient(model=model_metier,provider = provider, format_llm="json")  # adapte si tu utilises un autre modèle
        bot = BusinessClarificationBot(stats=compact_payload, llm=llm)

        print("=== Business Clarification Bot ===")

        # Première question
        question = bot.ask_next()
        print(question)

        while True:
            if '"Mode": "Question"' not in question:
                print("Fin de la session.")
                break
            
            user_answer = input("\nTa réponse > ")
            question = bot.ask_next(user_answer)
            print("\n" + question)

        # all_payload_short = apply_llm_business_annotations(compact_payload,question)
        all_payload_long = apply_llm_business_annotations(reports_stat,question)

        if print_json:

            # write_json.save_report_to_json(
            #     report=all_payload_short,
            #     output_path=f"Test/analyse/json/reponse_llm/short/test_analyse_metier_report_{nom}.json",
            # )
            write_json.save_report_to_json(
                report=all_payload_long,
                output_path=f"Test/analyse/{chemin_json}/all/test_analyse_metier_report_{nom}.json",
            )

        return all_payload_long
    else:
        return None
