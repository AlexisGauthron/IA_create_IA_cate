# Module Front

> Documentation technique complète du module `src/front/`

---

## Vue d'Ensemble

Le module **front** fournit une **interface Streamlit** pour le pipeline ML complet :

1. **pipeline_streamlit.py** : Pipeline ML complet (Analyse → FE → AutoML)

---

## Architecture du Module

```
src/front/
├── pipeline_streamlit.py       # Pipeline ML complet (7 étapes)
├── upload_fichier.py           # Utilitaires upload CSV
├── ui_helper.py                # Helpers Streamlit
├── css.py                      # Thème CSS personnalisé
├── fe_progress_monitor.py      # Suivi progression LLMFE
├── fe_runner_async.py          # Wrapper async pour LLMFE
└── components/
    ├── chat_component.py       # Composant chat réutilisable
    ├── llmfe_visualizer.py     # Dashboard résultats LLMFE
    └── results_component.py    # Affichage résultats
```

---

## Pipeline ML (pipeline_streamlit.py)

### Flow en 7 Étapes

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 0: UPLOAD                                                   │
│   • Charger train.csv (+ test.csv optionnel)                    │
│   • Détection auto séparateur CSV                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: CONFIGURATION                                            │
│   • Nom du projet                                               │
│   • Sélection colonne cible                                     │
│   • Type de tâche (classification/régression)                   │
│   • Config LLMFE (provider, model, iterations)                  │
│   • Config évaluation (métrique, modèles, agrégation)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: ANALYSE                                                  │
│   • Statistiques descriptives                                   │
│   • Corrélations features/target                                │
│   • Sauvegarde: outputs/{project}/analyse/stats_report.json     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: AGENT MÉTIER                                             │
│   • Chat avec BusinessClarificationBot                          │
│   • Questions sur le contexte métier                            │
│   • Recommandation de métrique                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: FEATURE ENGINEERING (LLMFE)                              │
│   • Exécution en arrière-plan (thread)                          │
│   • Suivi temps réel (polling JSON)                             │
│   • Dashboard métriques + graphique évolution                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: AUTOML                                                   │
│   • Sélection frameworks (AutoGluon, H2O, FLAML)                │
│   • Time budget                                                 │
│   • Entraînement et comparaison                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: RÉSULTATS                                                │
│   • Synthèse complète du pipeline                               │
│   • Export des résultats                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Lancement

```bash
# Activer l'environnement
conda activate Ia_create_ia

# Lancer l'interface
streamlit run src/front/pipeline_streamlit.py
```

### Gestion Session State

```python
st.session_state = {
    # Données
    "df_train": pd.DataFrame,
    "df_test": pd.DataFrame | None,
    "target_col": str,
    "project_name": str,

    # Progression
    "current_step": int,  # 0-6

    # Résultats
    "stats_report": dict,
    "correlations_df": pd.DataFrame,
    "fe_results": dict,
    "automl_results": dict,

    # Configuration LLM
    "agent_provider": "openai" | "ollama",
    "agent_model": str,
    "fe_provider": "openai" | "ollama",
    "fe_model": str,

    # LLMFE
    "max_fe_samples": int,
    "eval_metric": str,
    "eval_models": list[str],
    "eval_aggregation": str,
}
```

---

## Composants Réutilisables

### ChatComponent

```python
from src.front.components.chat_component import ChatComponent

chat = ChatComponent(
    session_key="chat_history",
    on_send=callback_function,
    commands={"skip": "Passer", "done": "Terminer"}
)
chat.render()
```

### AgentChatComponent

```python
from src.front.components.chat_component import AgentChatComponent

agent_chat = AgentChatComponent(
    session_key="agent_chat",
    provider="openai",
    model="gpt-4o-mini",
)
agent_chat.start_conversation(stats_report)
```

### LLMFEProgressTracker

```python
from src.front.components.llmfe_visualizer import LLMFEProgressTracker

tracker = LLMFEProgressTracker(max_samples=20)
tracker.update(sample_num=5, score=0.85, is_valid=True)
```

### LLMFEResultsDashboard

```python
from src.front.components.llmfe_visualizer import LLMFEResultsDashboard

dashboard = LLMFEResultsDashboard(results_dir="outputs/titanic/feature_engineering")
dashboard.render()
```

---

## Utilitaires

### Upload et Fusion CSV

```python
from src.front.upload_fichier import (
    read_csv_flex,           # Lecture avec détection séparateur
    merge_csv_files,         # Fusion multi-fichiers
    remove_duplicates,       # Suppression doublons
)

df = read_csv_flex(uploaded_file)
df_merged = merge_csv_files([df1, df2, df3])
df_clean = remove_duplicates(df_merged)
```

### Helpers UI

```python
from src.front.ui_helper import card_block

with card_block("Mon titre"):
    st.write("Contenu dans une card stylée")
```

### CSS Personnalisé

```python
from src.front.css import apply_custom_css

apply_custom_css()  # Applique le thème sombre avec cards
```

---

## Exécution Asynchrone LLMFE

### AsyncFERunner

```python
from src.front.fe_runner_async import AsyncFERunner

runner = AsyncFERunner(project_name="titanic")
runner.start(
    df_train=df,
    target_col="Survived",
    max_samples=20,
    api_model="gpt-4o",
)

# Polling
while runner.is_running():
    status = runner.get_status()
    time.sleep(2)

results = runner.get_results()
```

### FEProgressMonitor

```python
from src.front.fe_progress_monitor import FEProgressMonitor

monitor = FEProgressMonitor(samples_dir="outputs/titanic/.../samples")

progress = monitor.get_current_progress()
metrics = monitor.get_metrics()
chart_data = monitor.get_chart_data()
recent = monitor.get_recent_samples(n=5)
```

---

## Interactions avec Autres Modules

### Analyse

```python
from src.analyse.statistiques.report import analyze_dataset_for_fe
from src.analyse.correlation.correlation import FeatureCorrelationAnalyzer
from src.analyse.metier.chatbot_llm import BusinessClarificationBot
```

### Feature Engineering

```python
from src.feature_engineering.llmfe.llmfe_runner import LLMFERunner
from src.feature_engineering.path_config import FeatureEngineeringPathConfig
```

### AutoML

```python
from src.automl.runner import AutoMLRunner
```

### Core

```python
from src.core.llm_client import OllamaClient
from src.core.config import settings
```

---

## Structure des Outputs

```
outputs/{project_name}/
├── analyse/
│   ├── stats_report.json
│   ├── conversation.json
│   └── full_report.json
│
├── feature_engineering/
│   ├── samples/
│   │   ├── sample_1.json
│   │   └── ...
│   └── results/
│       ├── summary.json
│       ├── best_model.json
│       └── all_scores.json
│
└── automl/
    ├── models/
    └── results.json
```

---

## Voir Aussi

- [OVERVIEW.md](../architecture/OVERVIEW.md) - Vue d'ensemble du projet
- [pipeline.md](./pipeline.md) - Module pipeline (backend)
- [analyse.md](./analyse.md) - Module analyse
- [feature_engineering.md](./feature_engineering.md) - Module FE
