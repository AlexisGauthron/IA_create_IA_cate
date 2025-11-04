## Comparaisons AutoML 

# AutoML de ***FLAML*** (léger, rapide)

```py
automl.fit(
    X_train=self.X_train,  # EXEMPLES : np.ndarray (shape [n_samples, n_features]) | pd.DataFrame | scipy.sparse.csr_matrix/csc_matrix | list[list[float]]
    y_train=self.y_train,  # EXEMPLES : np.ndarray 1D | pd.Series | list[int/float/str] — binaire/multiclasse (int/str) ou régression (float)

    task=task,  # EXEMPLES : "classification" | "regression" | "rank" (learning-to-rank) | "ts_forecast" (séries temporelles)

    metric=metric,  # EXEMPLES :
                    #  - classification : "f1" | "accuracy" | "roc_auc" | "ap" (average_precision) | "log_loss" | "macro_f1" | "micro_f1" | "weighted_f1"
                    #  - regression     : "r2" | "mse" | "mae" | "rmse" | "mape" | "smape"
                    #  - rank           : "ndcg" | "map"
                    #  - ts_forecast    : "mse" | "mae" | "rmse" | "mape" | "smape"

    time_budget=time_budget,  # EXEMPLES (en secondes, int/float) : 30 | 60 | 300 (≈5 min) | 600 (10 min) | 1800 (30 min) | 3600 (1 h)

    log_file_name=f"{self.Nom_dossier}/flaml.log",  # EXEMPLES : "flaml.log" | "/tmp/flaml.log" | str(Path("logs")/"flaml_run.log") | None (pas de fichier)
                                                    # (chemin absolu/relatif accepté ; Path ou str)

    verbose=3,  # EXEMPLES : 0 (silencieux) | 1 (minimal) | 2 (normal) | 3 (détaillé/debug)

    mem_thres=taille_max_modele,  # EXEMPLES (octets par essai) : 0 (pas de plafond) | 1_500_000_000 (~1.5 Go) | 2_000_000_000 (2 Go décimal)
                                  #                              | 2*1024**3 (~2 GiB) | 3*1024**3 (~3 GiB)

    free_mem_ratio=espace_ram_libre,  # EXEMPLES (fraction de RAM à laisser libre) : 0.0 | 0.1 | 0.2 | 0.3 | 0.5  (plage typique : 0.0–0.8)
)

```

# AutoML de ***AutoGluon*** (très performant tabulaire)

```py
from autogluon.tabular import TabularPredictor

TabularPredictor(
    label="label",                # EXEMPLES : "label" (nom de la colonne cible)
                                 #  - Binaire/Multiclasse : int/str/catégories
                                 #  - Régression : float/int continus
    path=ag_dossier,             # EXEMPLES : "AutogluonModels/ag-001" | "./models" | None (répertoire par défaut)
    problem_type=None,           # EXEMPLES : None (Auto) | "binary" | "multiclass" | "regression"
    eval_metric=None,            # EXEMPLES :
                                 #   - classification : "accuracy" | "balanced_accuracy" | "f1" | "f1_macro" | "f1_weighted"
                                 #                      "roc_auc" | "log_loss" | "precision" | "recall" | "mcc"
                                 #   - regression     : "r2" | "rmse" | "mse" | "mae" | "mape" | "smape"
    verbosity=2,                 # EXEMPLES : 0 (silence) | 1 | 2 (par défaut) | 3 | 4 (très verbeux)
).fit(
    train_df,                    # EXEMPLES : pd.DataFrame | "train.csv" (chemin CSV)
                                 #  - Doit contenir la colonne cible `label`
                                 #  - Les NaN sont gérés automatiquement (imputation)

    tuning_data=None,            # EXEMPLES : None (split interne) | valid_df (DataFrame) | "valid.csv"
                                 #  - Si None, utiliser `holdout_frac` ci-dessous pour la taille de validation

    time_limit=300,              # EXEMPLES (secondes) : 30 | 120 | 300 (≈5 min) | 1800 (30 min) | 3600 (1 h)

    presets="medium_quality_faster_train",  # EXEMPLES (compromis prédéfinis) :
                                            #   "best_quality"                         (max qualité, plus lent)
                                            #   "high_quality_fast_inference_only_refit"
                                            #   "good_quality_faster_inference_only_refit"
                                            #   "medium_quality_faster_train"          (bon compromis)
                                            #   "optimize_for_deployment"              (inférence rapide, compact)
                                            #   "ignore_text"                          (ignore colonnes texte)
                                            #   "exploratory"                          (iteration rapide)

    hyperparameters="default",   # EXEMPLES :
                                 #   "default"                              (ensemble standard)
                                 #   {"GBM": {}}                            (LightGBM seul, params par défaut)
                                 #   {"GBM": {"num_boost_round": 1000},     # exemple LGBM
                                 #    "XGB": {"n_estimators": 600},         # exemple XGBoost
                                 #    "CAT": {"depth": 8},                  # exemple CatBoost
                                 #    "RF": {}, "XT": {}, "KNN": {}, "LR": {}, "NN_Torch": {}}
                                 #   []  (aucun modèle — rarement utile)
                                 #  Modèles tabulaires usuels : "GBM" (LightGBM), "CAT" (CatBoost), "XGB" (XGBoost),
                                 #  "RF" (RandomForest), "XT" (ExtraTrees), "KNN", "LR" (Logistic/Linear), "NN_Torch"

    hyperparameter_tune_kwargs=None,  # EXEMPLES :
                                      #   None                 (pas de HPO)
                                      #   "auto"               (HPO automatique)
                                      #   {"num_trials": 50, "scheduler": "local", "searcher": "auto"}
                                      #   {"num_trials": 200, "searcher": "bayesopt", "scheduler": "local"}

    excluded_model_types=None,  # EXEMPLES :
                                #   None
                                #   ["KNN", "NN_Torch"]           (exclure certains modèles)
                                #   ["RF", "XT", "LR"]

    num_bag_folds=0,            # EXEMPLES : 0 (pas de bagging) | 5 | 10 (meilleure robustesse, plus lent)
    num_bag_sets=1,             # EXEMPLES : 1 | 3 (répétitions de bagging)
    num_stack_levels=0,         # EXEMPLES : 0 (pas de stacking) | 1 | 2 (empilement multi-niveaux)

    holdout_frac=None,          # EXEMPLES : None (auto selon données) | 0.2 | 0.1
                                #  - Ignoré si `tuning_data` fourni

    sample_weight=None,         # EXEMPLES : None | "poids" (nom de colonne de poids dans train_df)

    calibration=None,           # EXEMPLES : None/"auto" | True | "sigmoid" | "isotonic"
                                #  - Pour probabilités mieux calibrées en classification

    refit_full=False,           # EXEMPLES : False | True
                                #  - True : réentraîne le(s) meilleur(s) modèle(s) sur tout le train (train+val)

    set_best_to_refit_full=False, # EXEMPLES : False | True
                                  #  - True : le modèle "best" pointe vers la version refit_full

    keep_only_best=False,       # EXEMPLES : False | True (supprime les modèles non meilleurs pour gagner de l’espace)
    save_space=False            # EXEMPLES : False | True (prune caches/fichiers pour réduire l’empreinte disque)
)

```

> Pas de check sur la taille du modèle 

#### Astuces pour forcer des modèles plus compacts avec AutoGluon
```py
presets = "optimize_for_deployment"  # tailles & latence réduites
```



# AutoML de ***TPOT*** (AutoML génétique scikit-learn) — checkpoints natifs

```py
TPOTClassifier(
    generations=7,                 # EXEMPLES (int ≥1) : 5 | 7 | 10 | 20 | 50
                                   #  - Plus de générations = meilleure recherche (souvent) mais plus long.

    population_size=25,            # EXEMPLES (int ≥2) : 20 | 25 | 50 | 100
                                   #  - Population plus grande = diversité accrue des pipelines, coût ↑.

    scoring="f1",                  # EXEMPLES (str sklearn ou callable make_scorer) :
                                   #  - Classification binaire : "f1" | "accuracy" | "precision" | "recall"
                                   #    "roc_auc" | "average_precision" | "neg_log_loss" | "balanced_accuracy" | "matthews_corrcoef"
                                   #  - Multiclasse : "f1_macro" | "f1_weighted" | "f1_micro" | "roc_auc_ovr" | "roc_auc_ovo"
                                   #  - (Callable) : from sklearn.metrics import make_scorer, fbeta_score
                                   #                scoring = make_scorer(fbeta_score, beta=2)
                                   #  - Régression -> utiliser TPOTRegressor (ex : "neg_mean_squared_error", "r2", "neg_mean_absolute_error")

    cv=5,                          # EXEMPLES : 5 | 10  (int = KFold/StratifiedKFold auto)
                                   #  - Objet CV : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                                   #               KFold(n_splits=5, shuffle=True, random_state=42)
                                   #               GroupKFold(n_splits=5)  # (passer `groups=` à .fit)
                                   #  - Pour données déséquilibrées => StratifiedKFold recommandé.

    verbosity=3,                   # EXEMPLES : 0 (silence) | 1 (info) | 2 (détails) | 3 (debug verbeux)

    periodic_checkpoint_folder="tpot_ckpt",  # EXEMPLES : None (désactivé) | "tpot_ckpt" | "./ckpt" | Path("ckpt_dir")
                                             #  - Sauvegarde périodique de l’état pour reprendre/inspecter.

    random_state=42,               # EXEMPLES : None (aléatoire) | 0 | 42 | np.random.RandomState(42)
                                   #  - Fixer pour la reproductibilité.

    n_jobs=-1                      # EXEMPLES : -1 (tous les cœurs) | 1 | 4 | 8
                                   #  - Plus de jobs = plus rapide mais RAM ↑.
)

```

# AutoML de ***H20 AutoML*** (leaderboard très lisible)

```py
H2OAutoML(
    max_runtime_secs=300,        # EXEMPLES (secondes) : 0 (pas de limite*) | 60 | 300 | 1800 | 3600
                                 # *Si ni `max_runtime_secs` ni `max_models` ne sont fournis, H2O règle dynamiquement ~1h par défaut.

    max_models=None,             # EXEMPLES : None | 10 | 25 | 100
                                 #  - Peut être combiné avec `max_runtime_secs` (arrêt au 1er seuil atteint).

    project_name="demo_automl",  # EXEMPLES : "credit_risk_v1" | "demo_automl" | None (nom auto-généré par H2O)

    seed=42,                     # EXEMPLES : None | 0 | 42 | 2025
                                 #  - Reproductible sous conditions (exclure "DeepLearning" pour une stricte réplicabilité).

    verbosity="info",            # EXEMPLES : None (désactive logs client) | "debug" | "info" | "warn"

    # =======================
    # ✨ Paramètres utiles en plus
    # =======================

    nfolds=-1,                   # EXEMPLES : -1 (auto: CV ou blending) | 0 (désactive CV & ensembles) | 3 | 5 | 10

    balance_classes=False,       # EXEMPLES : False | True (sur-/sous-échantillonnage pour classes rares)
    class_sampling_factors=None, # EXEMPLES : None | [1.0, 5.0, ...]  (nécessite balance_classes=True)
    max_after_balance_size=5.0,  # EXEMPLES : 0.5 | 1.0 | 2.0 | 5.0    (taille relative max après équilibrage)

    max_runtime_secs_per_model=0,# EXEMPLES : 0 (désactivé) | 10 | 60 | 120

    stopping_metric="AUTO",      # EXEMPLES : "AUTO" | "logloss" | "deviance" | "MSE" | "RMSE" | "MAE" | "RMSLE"
                                 #            "AUC" | "AUCPR" | "lift_top_group" | "misclassification" | "mean_per_class_error"

    stopping_tolerance=None,     # EXEMPLES : None (auto en fct taille n) | 1e-3 | 1e-4
    stopping_rounds=3,           # EXEMPLES : 0 (désactive) | 1 | 3 | 5 | 10

    sort_metric="AUTO",          # EXEMPLES : "AUTO" | "AUC" | "AUCPR" | "logloss" | "deviance" | "MSE" | "RMSE" | "MAE" | "RMSLE" | "mean_per_class_error"

    include_algos=None,          # EXEMPLES : None | ["GLM","GBM","DRF","XGBoost","DeepLearning","StackedEnsemble"]
    exclude_algos=None,          # EXEMPLES : None | ["DeepLearning","XGBoost"]  (⚠️ mutuellement exclusif avec include_algos)

    modeling_plan=None,          # EXEMPLES : None | liste d’étapes (avancé)
    preprocessing=None,          # EXEMPLES : None | ["target_encoding"] (expérimental)
    exploitation_ratio=0.0,      # EXEMPLES : 0.0 (désactivé) | 0.1 (reco si activé, expérimental)

    monotone_constraints=None,   # EXEMPLES : None | {"feature1": +1, "feature2": -1}

    keep_cross_validation_predictions=False,   # EXEMPLES : False | True (utile pour ré-ensembling)
    keep_cross_validation_models=False,        # EXEMPLES : False | True (consomme de la RAM)
    keep_cross_validation_fold_assignment=False,# EXEMPLES : False | True

    export_checkpoints_dir=None  # EXEMPLES : None | "checkpoints/" (export auto des modèles)
)
# ⚠️ Les données se passent à .train(), pas au constructeur :
# aml.train(y="label", x=None, training_frame=train_h2o,
#           validation_frame=valid_h2o, leaderboard_frame=lb_h2o,
#           blending_frame=blend_h2o, fold_column="kfold", weights_column="poids")
```


## Comparaisons 


| AutoML                  | **Temps disponible**                                                                            | **Taille du modèle**                                                                                                      | **Type de modèle exploré**                                                  | **Métrique d’évaluation**                                                                | **Feature engineering / déséquilibre**                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **FLAML**               | ✅ `time_budget`; limites par essai & latence prédiction (`train_time_limit`, `pred_time_limit`) | ✅ Contraintes mémoire (`mem_thres`, `free_mem_ratio`), possible pénaliser via métrique custom                             | ✅ `estimator_list` pour restreindre (LGBM, XGB, CatBoost, etc.)             | ✅ métrique string **ou** fonction custom; contraintes de métrique (`metric_constraints`) | 🟧 Basique intégré; **auto_augment** des classes rares, sampling stratifié; sinon prétraitement externe     |
| **AutoGluon (Tabular)** | ✅ `time_limit` (global), exécution séquentielle ou parallèle                                    | ✅ `memory_limit` (soft), **presets** d’optimisation (ex. `optimize_for_deployment`), contrainte d’inférence `infer_limit` | ✅ via `hyperparameters` / `included_model_types`                            | ✅ `eval_metric` (y compris métrique custom)                                              | ✅ Moteur de features (imputation, encodage, pruning), calibration du seuil; support robuste multi-modèles   |
| **TPOT**                | ✅ `max_time_mins` (global) + `max_eval_time_mins` (par pipeline)                                | 🟨 Pas de contrainte native sur la taille; possible **indirectement** via *scorer* custom qui pénalise la taille          | ✅ Espace de recherche configurable (`config_dict`) et gabarits (`template`) | ✅ `scoring` = nom sklearn **ou** *callable* (via `make_scorer`)                          | ✅ Optimise des **pipelines** (prétraitement, sélection de variables, construction de features)              |
| **H2O AutoML**          | ✅ `max_runtime_secs` (global) ou `max_models`; temps max par modèle                             | 🟨 Pas d’objectif “taille” direct; influence via choix d’algos / early-stopping                                           | ✅ `include_algos` / `exclude_algos` (GLM, GBM, XGBoost, DL, etc.)           | ✅ `stopping_metric` & `sort_metric` (AUC, AUCPR, logloss, RMSE, …)                       | ✅ Options natives pour **déséquilibre** (`balance_classes`, facteurs d’échantillonnage); prétraitement auto |


**https://chatgpt.com/s/dr_690a19a8f98c8191b0af98a53aaf9fe8**