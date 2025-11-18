import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler

# --- Data d'exemple ---
df = pd.DataFrame({
    "age": [25, 32, None, 40, 28],
    "revenu": [30000, 45000, 50000, None, 38000],
    "ville": ["Paris", "Lyon", "Lyon", "Marseille", None],
    "classe": [0, 1, 0, 1, 0],
})

X = df[["age", "revenu", "ville"]]
y = df["classe"]

# --- Pipeline Feature Engineering + modèle ---
fe_model = Pipeline(steps=[
    # 1) Imputation numérique
    ("impute_num", MeanMedianImputer(
        variables=["age", "revenu"], imputation_method="median"
    )),
    # 2) Encodage rare + One-hot sur ville
    ("rare_city", RareLabelEncoder(
        tol=0.1, n_categories=1, variables=["ville"]
    )),
    ("onehot_city", OneHotEncoder(
        variables=["ville"], drop_last=True
    )),
    # 3) Outliers sur revenu
    ("winsorize_revenu", Winsorizer(
        capping_method="gaussian", tail="both", fold=3, variables=["revenu"]
    )),
    # 4) Scaling
    ("scale", SklearnTransformerWrapper(
        transformer=StandardScaler(), variables=["age", "revenu"]
    )),
    # 5) Modèle
    ("clf", RandomForestClassifier(random_state=42))
])

fe_model.fit(X, y)
y_pred = fe_model.predict(X)
print("Prédiction FE + RF:", y_pred)
