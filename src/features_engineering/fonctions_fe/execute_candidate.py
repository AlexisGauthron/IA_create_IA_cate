from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer

# Assuming the FEGenerationConfig and generate_feature_space are defined as in the provided code.
# Paste them here if needed, but for brevity, assuming they are available.
from dataclasses import dataclass
@dataclass
class FEGenerationConfig:
    max_unary_numeric_per_col: int = 4
    max_pairwise_interactions: int = 50
    generate_family_size: bool = True   # ex: Titanic
    allow_text_embeddings: bool = True
    allow_text_tfidf: bool = False      # à activer si tu veux
    max_cat_for_one_hot: int = 10
    high_cardinality_threshold: int = 50


from src.features_engineering.fonctions_fe.generate_candidate import generate_feature_space


def apply_feature_engineering(
    df: pd.DataFrame,
    report: Dict[str, Any],
    config: Optional[FEGenerationConfig] = None,
) -> pd.DataFrame:
    """
    Applique les transformations de feature engineering générées à partir du rapport sur le DataFrame fourni.
    Ajoute de nouvelles colonnes au DataFrame en fonction des candidats générés.
    Gère les sorties multi-colonnes (ex: one-hot, embeddings) en utilisant des préfixes.
    
    Notes :
    - Pour le target encoding, utilise une validation croisée (5 folds) pour éviter les fuites de données.
    - Pour les embeddings texte, réduit à 32 dimensions via PCA pour éviter une explosion dimensionnelle.
    - Pour TF-IDF, limite à 50 features max.
    - Pour hashing, utilise 32 composants.
    - Assume que le target est numérique (pour mean encoding); adapté pour régression ou classification binaire.
    """
    if config is None:
        config = FEGenerationConfig()

    candidates = generate_feature_space(report, config)
    
    target_info = report["target"]
    target_name = target_info["name"]
    y = df[target_name]
    
    for i,cand in enumerate(candidates):
        name = cand["name"]
        typ = cand["type"]
        inputs = cand["inputs"]
        trans = cand.get("transformation", "")
        
        if typ == "numeric_derived":
            if "freq_encoding" in trans:
                col = inputs[0]
                freq = df[col].value_counts(normalize=True)
                df[name] = df[col].map(freq).fillna(0)
            
            elif len(inputs) == 1:
                col = inputs[0]
                if "log1p" in trans:
                    df[name] = np.log1p(df[col])
                elif "sqrt" in trans:
                    df[name] = np.sqrt(df[col])
                elif "square" in trans:
                    df[name] = df[col] ** 2
                elif "cube" in trans:
                    df[name] = df[col] ** 3
            
            elif len(inputs) == 2:
                c1, c2 = inputs
                if "prod" in trans:
                    df[name] = df[c1] * df[c2]
                elif "ratio" in trans:
                    df[name] = df[c1] / (df[c2] + 1e-6)
                elif "sum" in trans:
                    df[name] = df[c1] + df[c2]
            
            # Cas spécifique comme FamilySize
            if "FamilySize" in name and "SibSp" in inputs and "Parch" in inputs:
                df[name] = df["SibSp"] + df["Parch"] + 1
        
        elif typ == "categorical_encoding":
            col = inputs[0]
            encoding = cand.get("encoding")
            
            if encoding == "one_hot":
                dummies = pd.get_dummies(df[col], prefix=name, dtype=float)
                df = pd.concat([df, dummies], axis=1)
            
            elif encoding == "target_encoding":
                # Target encoding avec CV pour éviter leakage
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                df[name] = np.nan
                global_mean = y.mean()
                for train_idx, val_idx in kf.split(df):
                    train_df = df.iloc[train_idx]
                    val_df = df.iloc[val_idx]
                    means = train_df.groupby(col)[target_name].mean()
                    df.loc[val_df.index, name] = val_df[col].map(means).fillna(global_mean)
                df[name].fillna(global_mean, inplace=True)
            
            elif encoding == "hashing":
                # Hashing avec 32 composants
                hasher = FeatureHasher(n_features=32, input_type="string")
                hashed = hasher.fit_transform(df[[col]].astype(str).to_dict("records"))
                hashed_df = pd.DataFrame(hashed.toarray(), columns=[f"{name}_{i}" for i in range(32)], index=df.index)
                df = pd.concat([df, hashed_df], axis=1)
        
        elif typ == "text_representation":
            col = inputs[0]
            strategy = cand.get("text_strategy")
            
            if strategy == "embedding":
                model_name = cand.get("model", "intfloat/multilingual-e5-base")
                model = SentenceTransformer(model_name)
                embeds = model.encode(df[col].fillna("").tolist())
                # Réduction dimensionnelle à 32 via PCA
                pca = PCA(n_components=32)
                reduced_embeds = pca.fit_transform(embeds)
                embed_df = pd.DataFrame(reduced_embeds, columns=[f"{name}_{i}" for i in range(32)], index=df.index)
                df = pd.concat([df, embed_df], axis=1)
            
            elif strategy == "tfidf":
                vectorizer = TfidfVectorizer(max_features=50)
                tfidf_matrix = vectorizer.fit_transform(df[col].fillna(""))
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f"{name}_{feat}" for feat in vectorizer.get_feature_names_out()],
                    index=df.index
                )
                df = pd.concat([df, tfidf_df], axis=1)
        
        elif typ == "datetime_derived":
            col = inputs[0]
            # Assure que la colonne est datetime
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
            
            if "year" in trans:
                df[name] = df[col].dt.year
            elif "month" in trans:
                df[name] = df[col].dt.month
            elif "day" in trans:
                df[name] = df[col].dt.day
            elif "dayofweek" in trans:
                df[name] = df[col].dt.dayofweek

        print(f"Execution {cand} : n = {i}")
    return df