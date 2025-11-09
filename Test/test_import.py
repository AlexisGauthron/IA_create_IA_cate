from transformers import AutoModel, AutoTokenizer

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

libraries = [
    "torch",
    "numpy",
    "transformers",
    "pyarrow",
    "sentence_transformers",
    "tensorflow",
]


import tensorflow as tf
print(tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))


for lib in libraries:
    try:
        print(f"Import de {lib}...")
        __import__(lib)
        print(f"Import de {lib} réussi!\n")
    except Exception as e:
        print(f"Erreur lors de l'import de {lib}: {e}\n")



model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Modèle et tokenizer chargés avec succès !")


import streamlit as st
st.title("Test Streamlit sur Mac ✅")
st.write("Si tu vois ceci, Streamlit fonctionne correctement.")
if st.button("Clique moi"):
    st.success("Bouton cliqué !")
