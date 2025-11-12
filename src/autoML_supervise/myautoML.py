import time, json, pathlib, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import check_random_state
import joblib

import optuna
from optuna.pruners import MedianPruner

# (facultatif) XGBoost si installé
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

@dataclass
class AutoMLConfig:
    metric: str = "f1"
    n_splits: int = 5
    timeout_s: int = 600          # budget temps total
    n_trials: Optional[int] = None
    random_state: int = 42
    storage_uri: str = "sqlite:///autosimple.db"
    study_name: str = "autosimple"
    trials_log_path: str = "autosimple_trials.jsonl"
    best_model_path: str = "autosimple_best.joblib"

def split_features(X: pd.DataFrame) -> Tuple[list, list]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols

def make_pipeline(trial: optuna.Trial, X: pd.DataFrame) -> Pipeline:
    num_cols, cat_cols = split_features(X)

    estimator = trial.suggest_categorical("estimator", [
        "logreg", "rf", "hgb"] + (["xgb"] if HAS_XGB else [])
    )

    # Préproc: linéaire vs arbres
    if estimator == "logreg":
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])
        C = trial.suggest_float("logreg_C", 1e-3, 1e3, log=True)
        clf = LogisticRegression(max_iter=1000, C=C, n_jobs=None, random_state=0)
    elif estimator == "rf":
        pre = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ])
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 800, step=50),
            max_depth=trial.suggest_int("rf_max_depth", 3, 30),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("rf_max_features", 0.3, 1.0),
            n_jobs=-1, random_state=0
        )
    elif estimator == "hgb":
        pre = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ])
        clf = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("hgb_lr", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("hgb_max_depth", 2, 16),
            max_bins=trial.suggest_int("hgb_max_bins", 32, 255),
            l2_regularization=trial.suggest_float("hgb_l2", 1e-8, 1.0, log=True),
            early_stopping=True,
            random_state=0
        )
    else:  # xgb
        pre = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])
        clf = XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n", 200, 1200, step=100),
            max_depth=trial.suggest_int("xgb_depth", 3, 12),
            learning_rate=trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsamp", 0.5, 1.0),
            reg_lambda=trial.suggest_float("xgb_l2", 1e-8, 10.0, log=True),
            reg_alpha=trial.suggest_float("xgb_l1", 1e-8, 10.0, log=True),
            tree_method="hist",
            n_jobs=-1, random_state=0
        )

    return Pipeline([("pre", pre), ("clf", clf)])

def automl_fit(X: pd.DataFrame, y: pd.Series, cfg: AutoMLConfig):
    rng = check_random_state(cfg.random_state)
    scorer = make_scorer(f1_score)
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    pathlib.Path(cfg.trials_log_path).unlink(missing_ok=True)

    def objective(trial: optuna.Trial) -> float:
        start_t = time.time()
        pipe = make_pipeline(trial, X)

        scores = []
        for fold_idx, (tr, te) in enumerate(cv.split(X, y), 1):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            s = scorer._score_func(y.iloc[te], pred, **scorer._kwargs)
            scores.append(s)
            # Report à Optuna pour pruning
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
            # Garde un œil sur le temps global
            if cfg.timeout_s and (time.time() - study_start) > cfg.timeout_s:
                break

        mean_score = float(np.mean(scores))
        # Checkpoint JSONL
        with open(cfg.trials_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "number": trial.number,
                "value": mean_score,
                "params": trial.params,
                "folds": len(scores),
            }) + "\n")
        return mean_score

    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=cfg.storage_uri,
        load_if_exists=True,
        direction="maximize",
        pruner=MedianPruner(n_warmup_steps=2),
    )
    global study_start
    study_start = time.time()

    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        timeout=cfg.timeout_s if cfg.timeout_s else None,
        gc_after_trial=True,
        show_progress_bar=True
    )

    # Refit du meilleur pipeline sur tout le dataset
    best_params = study.best_trial.params
    best_pipe = make_pipeline(study.best_trial, X)
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, cfg.best_model_path)

    # Leaderboard synthétique
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df = df.sort_values("value", ascending=False)
    print("\n=== Leaderboard (top 10) ===")
    print(df.head(10).to_string(index=False))
    print(f"\nBest value: {study.best_value:.4f} | Best params: {best_params}")
    print(f"Saved: {cfg.best_model_path}  | Trials log: {cfg.trials_log_path}  | Storage: {cfg.storage_uri}")

    return study, best_pipe


# --- Demo: dataset jouet (sein) ---
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    cfg = AutoMLConfig(
        metric="f1",
        n_splits=5,
        timeout_s=300,      # 5 minutes de budget
        n_trials=None,      # illimité sous contrainte de temps
        random_state=42
    )
    automl_fit(X, pd.Series(y), cfg)
