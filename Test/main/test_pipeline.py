import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path
from sklearn.model_selection import train_test_split

import src.main.pipeline_model as pipe


if __name__ == "__main__":

    pipe.pipeline_create_model("Titanic_Kaggle","Survived")