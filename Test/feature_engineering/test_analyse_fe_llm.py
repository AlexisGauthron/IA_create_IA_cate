import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



from typing import Dict, List, Tuple, Optional
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.features_engineering.LLM_analyse_fe.pipeline import LLMFeatureEngineeringPipeline
from src.analyse.statistiques.config import FEAnalysisConfig
import pandas as pd

import src.Data.load_datasets as an


import requests

from src.helper.ollama_llm import OllamaClient

# On crée un client Ollama configuré comme on veut
ollama_client = OllamaClient(
    model="gpt-4.1-mini",                   # ou "llama3.1:8b", "deepseek-r1:8b", etc.
    base_url="http://localhost:11434/api/chat",
    provider = "openai",
    temperature=0,
    max_tokens=8024,
    format_llm="json",                     # équivalent à "format": "json" dans ta première fonction
)


def my_llm_func(prompt: str) -> str:
    """
    Wrapper simple pour adapter OllamaClient à l'interface attendue par la pipeline :
    une fonction qui prend un `prompt: str` et renvoie `str`.
    """
    messages = [
        # Tu peux ajouter un message system si tu veux contrôler le style :
        # {"role": "system", "content": "Tu es un assistant expert en feature engineering."},
        {"role": "user", "content": prompt},
    ]
    return ollama_client.chat(messages)




Nom_Projet = ["cate_metier","avis_client","Titanic_Kaggle","Verbatims"]
Label_Projet = ["label","label","Survived","Categorie"]

# Nom_Projet = ["Titanic_Kaggle"]
# Label_Projet = ["Survived"]

for (nom,label) in zip(Nom_Projet,Label_Projet):

    from src.features_engineering.helper.lire_json import load_json
    print("[INFO] Chargement Dataset\n")
    analyse = load_json(f"Test/feature_engineering/json/test_analyse_metier_report_{nom}.json")

    config = FEAnalysisConfig()
    pipeline = LLMFeatureEngineeringPipeline(config=config)

    result = pipeline.analyse_and_plan(
        stats=analyse,
        llm_func=my_llm_func,
        user_description="Dataset de tickets clients avec catégorie cible.",
        extra_instructions="Privilégie des features simples et interprétables.",
        print_prompt=True,  # pour voir le prompt généré
    )

    report = result["report"]
    prompt = result["prompt"]
    plan = result["plan"]

    print("Notes globales proposées par le LLM :")
    for note in plan.global_notes:
        print(" -", note)

    print("\nFeatures proposées :")
    for spec in plan.features_plan:
        print(f" - {spec.name} ({spec.type}) à partir de {spec.inputs} -> {spec.reason}")
        print(f"Transformation : {spec.transformation}\n Descriptions : {spec.descriptions_transformations}\n")

    print("\nQuestions à poser à l'utilisateur :")
    for q in plan.questions_for_user:
        print(" ?", q)

    print(f"\n[DEBUG] Result : {plan}\n\n")

