# llmfe_runner.py
"""
Point d'entrée simplifié pour LLMFE.
Utilise FeatureEngineeringPathConfig pour une architecture simplifiée.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.feature_engineering.path_config import FeatureEngineeringPathConfig
from src.feature_engineering.llmfe import config as config_lib
from src.feature_engineering.llmfe import pipeline
from src.feature_engineering.llmfe import sampler
from src.feature_engineering.llmfe import evaluator
from src.feature_engineering.llmfe.feature_formatter import FeatureFormat
from src.feature_engineering.llmfe.feature_insights import FeatureInsights


class LLMFERunner:
    """
    Runner simplifié pour exécuter LLMFE sur un dataset.

    Exemple d'utilisation:
    ```python
    runner = LLMFERunner(project_name="titanic")

    result = runner.run(
        df_train=df,
        target_col="Survived",
        is_regression=False,
        max_samples=20
    )
    ```
    """

    def __init__(
        self,
        project_name: str,
        output_root: Optional[str] = None,
        path_config: Optional[FeatureEngineeringPathConfig] = None,
    ):
        """
        Initialise le runner LLMFE.

        Args:
            project_name: Nom du projet (définit le dossier de sortie)
            output_root: Racine des dossiers de sortie (utilise Settings si None)
            path_config: PathConfig existant (si fourni, ignore les autres params)
        """
        self.project_name = project_name
        self.output_root = output_root

        # Utiliser le path_config fourni ou en créer un nouveau
        if path_config is not None:
            self.path_config = path_config
        else:
            self.path_config = None  # Sera créé dans run()

    def run(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        is_regression: bool = False,
        max_samples: int = 20,
        task_description: Optional[str] = None,
        meta_data: Optional[Dict[str, str]] = None,
        use_api: bool = True,
        api_model: str = "gpt-4",
        num_samplers: int = 1,
        num_evaluators: int = 1,
        samples_per_prompt: int = 3,
        evaluate_timeout_seconds: int = 30,
        feature_insights: Optional[FeatureInsights] = None,
        feature_format: FeatureFormat = FeatureFormat.BASIC,
        analyse_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Exécute LLMFE sur le dataset fourni.

        Args:
            df_train: DataFrame d'entraînement
            target_col: Nom de la colonne cible
            is_regression: True si régression, False si classification
            max_samples: Nombre maximum d'itérations LLM
            task_description: Description de la tâche (auto-générée si None)
            meta_data: Dictionnaire des métadonnées des features
            use_api: Utiliser l'API OpenAI
            api_model: Modèle API à utiliser
            num_samplers: Nombre de samplers parallèles
            num_evaluators: Nombre d'évaluateurs parallèles
            samples_per_prompt: Nombre de samples par prompt
            evaluate_timeout_seconds: Timeout d'évaluation
            feature_insights: FeatureInsights pré-calculés (optionnel)
            feature_format: Format des features (BASIC, TAGS, HIERARCHICAL)
            analyse_path: Chemin vers le JSON d'analyse existant (optionnel)

        Returns:
            Dictionnaire avec les résultats et chemins
        """
        # 1. Créer la configuration des chemins si pas déjà fournie
        if self.path_config is None:
            self.path_config = FeatureEngineeringPathConfig(
                project_name=self.project_name,
                base_dir=self.output_root,
            )

        # Afficher la configuration
        self._print_config()

        # 2. Générer la description de la tâche si non fournie
        if task_description is None:
            task_description = f"Predict '{target_col}' from the given features."

        # 3. Générer et sauvegarder la spec
        spec_content = self._generate_spec(
            task_description=task_description,
            is_regression=is_regression,
        )
        spec_path = self.path_config.save_spec(spec_content)
        print(f"✅ Spec générée: {spec_path}")

        # 4. Préparer les données
        X = df_train.drop(columns=[target_col])
        y = df_train[target_col].values

        # Déterminer les colonnes catégorielles
        is_cat = [X[col].dtype == 'object' or X[col].dtype.name == 'category'
                  for col in X.columns]

        data_dict = {
            'inputs': X,
            'outputs': y,
            'is_cat': is_cat,
            'is_regression': is_regression,
        }
        dataset = {'data': data_dict}

        # 5. Créer les métadonnées si non fournies
        if meta_data is None:
            meta_data = {col: col.replace('_', ' ') for col in X.columns}

        # 5b. Charger ou créer les FeatureInsights
        # IMPORTANT: On utilise TOUJOURS src/analyse/ comme source unique de vérité
        if feature_insights is None and analyse_path is not None:
            # Charger depuis le JSON d'analyse existant
            try:
                feature_insights = FeatureInsights.from_json(analyse_path)
                print(f"✅ Insights chargés depuis: {analyse_path}")
            except FileNotFoundError:
                print(f"⚠️ Fichier d'analyse non trouvé: {analyse_path}")
                print("   → Lancement de l'analyse via src/analyse/...")
                feature_insights = FeatureInsights.from_analyse(
                    df=df_train,
                    target_col=target_col,
                    project_name=self.project_name,
                    compute_correlations=True,
                )
        elif feature_insights is None and feature_format != FeatureFormat.BASIC:
            # Lancer l'analyse automatiquement si format enrichi demandé
            print("📊 Lancement de l'analyse via src/analyse/...")
            feature_insights = FeatureInsights.from_analyse(
                df=df_train,
                target_col=target_col,
                project_name=self.project_name,
                compute_correlations=True,
            )

        # Afficher le format utilisé
        print(f"📝 Format des features: {feature_format.value}")

        # 6. Configurer LLMFE
        class_config = config_lib.ClassConfig(
            llm_class=sampler.LocalLLM,
            sandbox_class=evaluator.LocalSandbox,
        )

        config = config_lib.Config(
            use_api=use_api,
            api_model=api_model,
            num_samplers=num_samplers,
            num_evaluators=num_evaluators,
            samples_per_prompt=samples_per_prompt,
            evaluate_timeout_seconds=evaluate_timeout_seconds,
        )

        # 7. Lancer la pipeline
        print("\n" + "=" * 60)
        print("           DÉMARRAGE DE LLMFE")
        print("=" * 60)
        print(f"  Projet      : {self.project_name}")
        print(f"  Target      : {target_col}")
        print(f"  Type        : {'Régression' if is_regression else 'Classification'}")
        print(f"  Max samples : {max_samples}")
        print(f"  Modèle API  : {api_model}")
        print("=" * 60 + "\n")

        specification = self.path_config.read_spec()

        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config,
            meta_data=meta_data,
            max_sample_nums=max_samples,
            class_config=class_config,
            path_config=self.path_config,
            feature_insights=feature_insights,
            feature_format=feature_format,
            target_column=target_col,
        )

        # 8. Retourner les résultats
        return {
            "path_config": self.path_config,
            "project_dir": str(self.path_config.project_dir),
            "results_dir": str(self.path_config.llmfe_results_dir),
            "samples_dir": str(self.path_config.llmfe_samples_dir),
        }

    def _print_config(self):
        """Affiche la configuration des chemins."""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  LLMFE Path Configuration (Simplified)                       ║
╠══════════════════════════════════════════════════════════════╣
║  Project   : {self.project_name:<47} ║
║  Timestamp : {self.path_config.timestamp:<47} ║
╠══════════════════════════════════════════════════════════════╣
║  Output Structure:                                           ║
║  {str(self.path_config.project_dir):<58} ║
║  ├── features/      (train_fe.parquet, test_fe.parquet)      ║
║  ├── llmfe/                                                  ║
║  │   ├── samples/   (JSON samples)                           ║
║  │   ├── results/   (best_model, scores, report)             ║
║  │   └── tensorboard/                                        ║
║  ├── specs/         (specification.txt)                      ║
║  ├── transforms/    (pipeline.pkl)                           ║
║  └── logs/                                                   ║
╚══════════════════════════════════════════════════════════════╝
""")

    def _generate_spec(self, task_description: str, is_regression: bool) -> str:
        """Génère une spec dynamiquement selon le type de problème."""

        if is_regression:
            model = "xgb.XGBRegressor"
            metric_import = "from sklearn.metrics import mean_squared_error"
            score_calc = "score = -1 * mean_squared_error(y_test, y_pred, squared=False)"
            y_transform = "y = outputs"
            kfold = "kf = KFold(n_splits=4, shuffle=True, random_state=42)"
        else:
            model = "xgb.XGBClassifier"
            metric_import = "from sklearn.metrics import accuracy_score"
            score_calc = "score = accuracy_score(y_test, y_pred)"
            y_transform = "y = label_encoder.fit_transform(outputs)"
            kfold = "kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)"

        return f'''"""
[PREFIX]

###
<Task>
{task_description}

###
<Features>
[FEATURES]

###
<Examples>
[EXAMPLES]
[SUFFIX]
"""

@evaluate.run
def evaluate(data: dict):
    """Evaluate the feature transformations on data observations."""
    from sklearn import preprocessing
    from sklearn.model_selection import StratifiedKFold, KFold
    {metric_import}
    from src.feature_engineering.llmfe.preprocessing import preprocess_datasets
    import xgboost as xgb
    import numpy as np

    label_encoder = preprocessing.LabelEncoder()
    inputs, outputs, is_cat, is_regression = data['inputs'], data['outputs'], data['is_cat'], data['is_regression']
    X = modify_features(inputs)
    {y_transform}

    # Encode categorical string columns
    for col in X.columns:
        if X[col].dtype == 'string' or X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    {kfold}
    scores = []

    # 4-Fold Cross-Validation
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_new, X_test_new = preprocess_datasets(X_train, X_test, None)

        model = {model}(random_state=42)
        model.fit(X_train_new, y_train)
        y_pred = model.predict(X_test_new)
        {score_calc}
        scores.append(score)

    return np.mean(scores), inputs, outputs


@equation.evolve
def modify_features(df_input) -> pd.DataFrame:
    """
    Initial feature engineering function.
    This function will be evolved by the LLM to create better features.
    """
    import pandas as pd
    import numpy as np

    df_output = df_input.copy()
    return df_output
'''


def run_llmfe(
    project_name: str,
    df_train: pd.DataFrame,
    target_col: str,
    is_regression: bool = False,
    max_samples: int = 20,
    output_root: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fonction raccourcie pour lancer LLMFE.

    Args:
        project_name: Nom du projet
        df_train: DataFrame d'entraînement
        target_col: Colonne cible
        is_regression: True si régression
        max_samples: Nombre max d'itérations
        output_root: Racine de sortie (utilise Settings si None)
        **kwargs: Arguments additionnels passés à LLMFERunner.run()

    Returns:
        Dictionnaire avec les résultats

    Exemple:
    ```python
    result = run_llmfe(
        project_name="MonProjet",
        df_train=df,
        target_col="target",
        is_regression=False,
        max_samples=20
    )
    ```
    """
    runner = LLMFERunner(
        project_name=project_name,
        output_root=output_root,
    )

    return runner.run(
        df_train=df_train,
        target_col=target_col,
        is_regression=is_regression,
        max_samples=max_samples,
        **kwargs
    )
