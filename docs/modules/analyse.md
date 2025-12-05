# Module Analyse

> Documentation technique complète du module `src/analyse/`

---

## Vue d'Ensemble

Le module **analyse** est responsable de :
1. **Profilage statistique** : Analyser la structure, les types et les distributions du dataset
2. **Détection automatique** : Identifier le type de problème (classification/régression), les fuites de données, les anomalies
3. **Enrichissement LLM** : Dialogue interactif pour comprendre le contexte métier et recommander la métrique optimale

---

## Architecture du Module

```
src/analyse/
├── analyse.py                    # Point d'entrée principal
├── path_config.py                # Gestion des chemins (hérite BasePathConfig)
│
├── dataset/                      # Dataclasses pour structures LLM
│   ├── all.py                    # FEDatasetSnapshotForLLM (classe principale)
│   ├── cibles.py                 # TargetSummaryForLLM
│   ├── contextes.py              # DatasetContextForLLM
│   ├── features.py               # FeatureSummaryForLLM
│   ├── globale.py                # BasicDatasetStats
│   ├── leakage.py                # LeakageSignalForLLM
│   └── type_stats.py             # NumericStats, CategoricalStats, TextStats
│
├── statistiques/                 # Analyse statistique
│   ├── config.py                 # FEAnalysisConfig (seuils configurables)
│   ├── report.py                 # analyze_dataset_for_fe() - fonction centrale
│   ├── targets.py                # Analyse des colonnes cibles
│   ├── features.py               # Analyse des features
│   ├── leakage.py                # Détection de fuites
│   ├── features_types.py         # Helpers pour stats typées
│   ├── printing.py               # Affichage rapport texte
│   └── write_json.py             # Sauvegarde JSON
│
├── correlation/                  # Corrélations avancées
│   └── correlation.py            # FeatureCorrelationAnalyzer
│
├── metier/                       # Interaction LLM
│   ├── chatbot_llm.py            # BusinessClarificationBot
│   ├── prompt_metier.py          # Prompts système
│   ├── parsing_json.py           # Fusion réponses LLM
│   └── business_agent.py         # Orchestration session LLM
│
└── helper/                       # Utilitaires
    ├── compress_data.py          # Réduction tokens
    ├── helper_json_safe.py       # Sérialisation JSON-safe
    └── suppression_vnul.py       # Nettoyage nulls
```

---

## Flux de Données

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ENTRÉE: DataFrame                            │
│                  + target_cols + nom_projet                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. ANALYSE STATISTIQUE (analyze_dataset_for_fe)                      │
│    ├── analyze_targets() → Détecte problem_type                      │
│    ├── analyze_features() → Rôle et flags de chaque feature          │
│    ├── detect_leakage() → Corrélations > 0.97                        │
│    └── [optionnel] compute_correlations() → Pearson, MI, PhiK        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SORTIE INTERMÉDIAIRE: llm_payload (dict JSON)                        │
│   - context, basic_stats, target, features, leakage_signals          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (si only_stats=False)
┌─────────────────────────────────────────────────────────────────────┐
│ 2. ENRICHISSEMENT LLM                                                │
│    ├── compact_llm_snapshot_payload() → Réduit tokens                │
│    ├── BusinessClarificationBot() → Dialogue interactif              │
│    └── apply_llm_business_annotations() → Fusionne réponses          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. SAUVEGARDE                                                        │
│    ├── {output}/stats/report_stats.json (sans LLM)                   │
│    ├── {output}/full/report_full.json (avec LLM)                     │
│    └── {output}/agent_llm/conversation.json                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Classes et Fonctions Principales

### 1. Point d'Entrée : `analyse.py`

```python
def analyse(
    df: pd.DataFrame,
    target_cols: str | list[str],
    nom: str,
    print_json: bool = False,
    chemin_json: str = "json",
    provider: str = "ollama",
    model_metier: str = "deepseek-r1:8b",
    only_stats: bool = False,
    with_correlations: bool = False,
    ...
) -> dict | None:
    """
    Orchestre l'analyse complète du dataset.

    Args:
        df: DataFrame à analyser
        target_cols: Colonne(s) cible
        nom: Nom du projet
        print_json: Si True, sauvegarde les rapports JSON
        provider: "ollama" ou "openai"
        model_metier: Modèle LLM pour enrichissement métier
        only_stats: Si True, pas d'enrichissement LLM
        with_correlations: Active le calcul des corrélations avancées

    Returns:
        llm_payload enrichi ou None
    """
```

### 2. Gestion des Chemins : `path_config.py`

```python
class AnalysePathConfig(BasePathConfig):
    """Gestion des chemins de sortie pour l'analyse."""

    MODULE_NAME = "analyse"

    # Attributs créés automatiquement
    stats_dir: Path      # → outputs/{projet}/analyse/stats/
    full_dir: Path       # → outputs/{projet}/analyse/full/
    agent_llm_dir: Path  # → outputs/{projet}/analyse/agent_llm/

    # Propriétés
    @property
    def stats_report_path(self) -> Path:
        """Chemin du rapport statistique."""
        return self.stats_dir / "report_stats.json"

    @property
    def full_report_path(self) -> Path:
        """Chemin du rapport complet (avec LLM)."""
        return self.full_dir / "report_full.json"

    # Méthodes
    def save_stats_report(self, report: dict) -> Path: ...
    def save_full_report(self, report: dict) -> Path: ...
    def save_conversation(self, data: dict) -> Path: ...
    def has_llm_analysis(self) -> bool: ...
```

### 3. Analyse Statistique : `statistiques/report.py`

```python
def analyze_dataset_for_fe(
    df: pd.DataFrame,
    target_cols: str | list[str],
    config: FEAnalysisConfig | None = None,
    print_report: bool = True,
    dataset_name: str = "dataset",
    with_correlations: bool = False,
) -> dict:
    """
    Analyse complète du dataset pour Feature Engineering.

    Returns:
        {
            "global": {n_rows, n_features, ...},
            "targets": {target_name: stats},
            "features": {col_name: stats},
            "suspected_leakage": [...],
            "warnings": [...],
            "llm_snapshot": FEDatasetSnapshotForLLM,
            "llm_payload": dict,
            "correlations": dict (si with_correlations)
        }
    """
```

### 4. Configuration : `statistiques/config.py`

```python
@dataclass
class FEAnalysisConfig:
    """Configuration des seuils d'analyse."""

    max_classes_for_classif: int = 20     # Seuil classification/régression
    max_unique_cat_low: int = 20          # Cardinalité basse catégorielle
    high_cardinality_threshold: int = 50  # Flag HIGH_CARDINALITY
    text_unique_ratio_threshold: float = 0.5  # Détecte texte libre
    id_unique_ratio_threshold: float = 0.9    # Détecte ID-like
    high_missing_threshold: float = 0.3       # % NaN critiques
    strong_corr_threshold: float = 0.97       # Détecte leakage
```

### 5. Corrélations : `correlation/correlation.py`

```python
class FeatureCorrelationAnalyzer:
    """Analyse complète des corrélations."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        task: str = "classification"
    ): ...

    def compute_classical_correlations(self) -> dict:
        """Pearson, Spearman, Kendall."""

    def compute_mutual_info(self) -> pd.Series:
        """Information mutuelle."""

    def compute_mic_matrix(self) -> pd.DataFrame | None:
        """MIC (si minepy installé)."""

    def compute_phik(self) -> pd.DataFrame | None:
        """PhiK (si phik installé)."""

    def combined_feature_score(self, normalize: bool = True) -> pd.DataFrame:
        """Score combiné de toutes les méthodes."""
```

### 6. Dialogue LLM : `metier/chatbot_llm.py`

```python
@dataclass
class BusinessClarificationBot:
    """Gère le dialogue avec le LLM pour clarification métier."""

    stats: dict | str           # Snapshot JSON du dataset
    llm: OllamaClient           # Client LLM
    messages: list[dict]        # Historique conversation

    def ask_next(self, user_answer: str | None = None) -> str:
        """
        Pose la prochaine question ou traite la réponse.

        Args:
            user_answer: Réponse de l'utilisateur (None pour démarrer)

        Returns:
            Réponse du LLM (question ou analyse finale)
        """
```

### 7. Fusion LLM : `metier/parsing_json.py`

```python
def apply_llm_business_annotations(
    snapshot: dict,
    llm_result: str,
    min_confidence: float = 0.6
) -> dict:
    """
    Fusionne les réponses LLM dans le snapshot.

    Enrichit:
    - context.business_description
    - context.final_metric + final_metric_reason
    - features[*].feature_description

    Args:
        snapshot: Payload llm_payload original
        llm_result: Réponse JSON du LLM
        min_confidence: Seuil de confiance minimum (0-1)

    Returns:
        Snapshot enrichi avec annotations LLM
    """
```

---

## Dataclasses (structures de données)

### FEDatasetSnapshotForLLM (classe principale)

```python
@dataclass
class FEDatasetSnapshotForLLM:
    """Snapshot complet du dataset pour le LLM."""

    context: DatasetContextForLLM
    basic_stats: BasicDatasetStats
    target: TargetSummaryForLLM
    features: list[FeatureSummaryForLLM]
    leakage_signals: list[LeakageSignalForLLM]
    analysis_config: dict[str, Any]
    global_notes: list[str]

    def to_llm_payload(self) -> dict:
        """Convertit en dict JSON-serializable."""
```

### TargetSummaryForLLM

```python
@dataclass
class TargetSummaryForLLM:
    name: str
    problem_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
        "high_cardinality_classification"
    ]
    class_counts: dict[str, int] | None
    class_proportions: dict[str, float] | None
    imbalance_ratio: float | None
    is_imbalanced: bool | None
    positive_class: str | None  # Pour classification binaire
    notes: list[str]
```

### FeatureSummaryForLLM

```python
@dataclass
class FeatureSummaryForLLM:
    name: str
    role: Literal["feature", "target", "id", "timestamp", "group", "text"]
    inferred_type: Literal[
        "numeric", "categorical_low", "categorical_high",
        "text", "datetime", "bool", "constant", "id_like", "unknown"
    ]
    n_missing: int
    missing_rate: float
    n_unique: int
    unique_ratio: float
    example_values: list[str]
    numeric_stats: NumericStats | None
    categorical_stats: CategoricalStats | None
    text_stats: TextStats | None
    flags: list[str]      # ["CONSTANT", "ID_LIKE", "HIGH_CARDINALITY"]
    notes: list[str]
    fe_hints: list[str]   # Suggestions Feature Engineering
    feature_description: str | None  # Enrichi par LLM
```

---

## Logique de Détection

### Détection du Type de Problème

```
Numérique:
  ├── n_unique <= 2 → "binary_classification"
  ├── n_unique <= max_classes AND ratio < 20% → "multiclass_classification"
  └── Sinon → "regression"

Catégorique:
  ├── n_unique <= 2 → "binary_classification"
  ├── n_unique <= max_classes → "multiclass_classification"
  └── Sinon → "high_cardinality_classification"
```

### Détection du Rôle d'une Feature

```
1. datetime? → "datetime"
2. booléen (True/False)? → "boolean"
3. numérique non-bool? → "numeric"
4. object/categorical:
   ├── (n_unique > 20) AND (unique_ratio > 0.5)? → "text"
   └── Sinon → "categorical"
5. Sinon → "unknown"
```

### Flags Détectés

| Flag | Condition |
|------|-----------|
| `CONSTANT` | n_unique <= 1 |
| `ID_LIKE` | nom contient "id/uuid" OU unique_ratio > 0.9 |
| `HIGH_CARDINALITY` | catégorielle ET n_unique > 50 |

### Suggestions Feature Engineering

| Type | Suggestions |
|------|-------------|
| Numérique | Scaling, log-transform, binning, outlier handling |
| Catégorielle low | One-hot encoding |
| Catégorielle high | Target encoding, hashing |
| Texte | TF-IDF, embeddings, n-grams |
| DateTime | Extraction (year, month, day, weekday, hour) |

---

## Exemple de Payload JSON

```json
{
  "context": {
    "name": "titanic",
    "business_description": null,
    "metric": null,
    "is_time_dependent": false
  },
  "basic_stats": {
    "n_rows": 891,
    "n_columns": 12,
    "n_features": 11,
    "missing_cell_ratio": 0.0178
  },
  "target": {
    "name": "Survived",
    "problem_type": "binary_classification",
    "class_counts": {"0": 549, "1": 342},
    "imbalance_ratio": 1.604,
    "is_imbalanced": false
  },
  "features": [
    {
      "name": "Age",
      "role": "feature",
      "inferred_type": "numeric",
      "missing_rate": 0.1986,
      "numeric_stats": {
        "mean": 29.70,
        "std": 14.53,
        "skewness": 0.39
      },
      "fe_hints": ["numeric_imputation", "candidate_for_scaling"]
    },
    {
      "name": "Cabin",
      "role": "feature",
      "inferred_type": "categorical_high",
      "missing_rate": 0.7709,
      "n_unique": 147,
      "flags": ["HIGH_CARDINALITY"],
      "fe_hints": ["use_target_encoding_or_hashing"]
    }
  ],
  "leakage_signals": []
}
```

---

## Utilisation par les Autres Modules

### Pipeline (`src/pipeline/pipeline_all.py`)

```python
from src.analyse.statistiques.report import analyze_dataset_for_fe
from src.analyse.path_config import AnalysePathConfig
from src.analyse.metier.chatbot_llm import BusinessClarificationBot

# Analyse statistique
report = analyze_dataset_for_fe(df, target_cols, with_correlations=True)
llm_payload = report["llm_payload"]

# Sauvegarde
paths = AnalysePathConfig(project_name)
paths.save_stats_report(llm_payload)

# Enrichissement LLM (optionnel)
bot = BusinessClarificationBot(stats=llm_payload, llm=llm_client)
response = bot.ask_next()
```

### Feature Engineering (`src/feature_engineering/`)

```python
from src.analyse.path_config import AnalysePathConfig

# Récupérer le rapport d'analyse existant
paths = AnalysePathConfig(project_name)
if paths.has_llm_analysis():
    report = paths.load_full_report()
    metric = report["context"].get("final_metric", "f1")
```

### Frontend (`src/front/pipeline_streamlit.py`)

```python
from src.analyse.statistiques.report import analyze_dataset_for_fe

# Analyse pour affichage
report = analyze_dataset_for_fe(df, target_col, print_report=False)
st.json(report["llm_payload"])
```

---

## Voir Aussi

- [OVERVIEW.md](../architecture/OVERVIEW.md) - Vue d'ensemble du projet
- [MODULE_DEPENDENCIES.md](../architecture/MODULE_DEPENDENCIES.md) - Dépendances entre modules
- [core.md](./core.md) - Module core (BasePathConfig, OllamaClient)
