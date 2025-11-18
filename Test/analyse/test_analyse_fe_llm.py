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

from src.features_engineering.pipeline import LLMFeatureEngineeringPipeline
from src.analyse.statistiques.config import FEAnalysisConfig
import pandas as pd

import src.Data.load_datasets as an


import requests

def ollama_mistral_llm(
    prompt: str,
    model: str = "mistral",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """
    Appelle le modèle Mistral via Ollama et renvoie uniquement le texte généré.

    Paramètres
    ----------
    prompt : str
        Le texte à envoyer au modèle.
    model : str
        Nom du modèle Ollama (ex: "mistral", "mistral:instruct", "mistral-nemo", etc.).
    base_url : str
        URL de base du serveur Ollama (par défaut localhost).
    timeout : int
        Timeout en secondes pour la requête HTTP.

    Retour
    ------
    response_text : str
        Le contenu texte renvoyé par le modèle (champ `message.content`).
    """
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,  # on veut tout le texte d'un coup
        "format": "json"
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Format de réponse typique d'Ollama /api/chat :
    # {
    #   "model": "...",
    #   "created_at": "...",
    #   "message": {"role": "assistant", "content": "..."},
    #   ...
    # }
    return data["message"]["content"]



def my_llm_func(prompt: str) -> str:
    return ollama_mistral_llm(prompt, model="mistral:latest")  # ou "mistral:instruct"






Nom_Projet = ["cate_metier","avis_client","Titanic_Kaggle","Verbatims"]
Label_Projet = ["label","label","Survived","Categorie"]

Nom_Projet = ["Titanic_Kaggle"]
Label_Projet = ["Survived"]

for (nom,label) in zip(Nom_Projet,Label_Projet):

    print("[INFO] Chargement Dataset\n")
    # Chargement dataset
    try:
        df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{nom}")
    except:
        df_train, df_test = an.csv_to_dataframe_train_test(f"Data/{nom}", sep=";")


    config = FEAnalysisConfig()
    pipeline = LLMFeatureEngineeringPipeline(config=config)

    result = pipeline.analyse_and_plan(
        df_train,
        target_cols=label,
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

    print("\nQuestions à poser à l'utilisateur :")
    for q in plan.questions_for_user:
        print(" ?", q)

