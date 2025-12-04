import os
import sys
import warnings

# Supprime les warnings non critiques
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.pipeline.pipeline_autoMl import pipeline_create_model

if __name__ == "__main__":
    model_autoML = ["flaml", "autogluon", "tpot", "h2o"]

    # Pour un test rapide, on utilise seulement h2o (le plus rapide)
    model = ["h2o"]

    # Titanic dataset avec colonne cible "Survived"
    pipeline_create_model("titanic", "Survived", autoML=model)
