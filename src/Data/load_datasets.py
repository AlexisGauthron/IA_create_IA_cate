import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits



def load_datasets_breast_cancer():
    print("[INFO] Load datasets\n")
    ds = load_breast_cancer(as_frame=True) 
    df = ds.frame   
    return df

def load_datasets_iris():
    print("[INFO] Load datasets\n")
    ds = load_iris(as_frame=True)
    df = ds.frame
    return df

def load_datasets_wine():
    print("[INFO] Load datasets\n")
    ds = load_wine(as_frame=True)
    df = ds.frame 
    return df

def load_datasets_digits():
    print("[INFO] Load datasets\n")
    ds = load_digits(as_frame=True)
    df = ds.frame 
    return df