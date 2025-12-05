# Dépendances entre Modules

> Ce document explique comment le module `core/` est utilisé par les autres modules,
> et comment créer un nouveau module qui s'intègre proprement.

---

## Vue d'Ensemble des Dépendances

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              src/core/                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  config.py ─────────────► Utilisé par TOUS les modules                     │
│  (settings)               pour accéder aux clés API et chemins             │
│                                                                            │
│  base_path_config.py ───► Hérité par analyse/, feature_engineering/,       │
│  (BasePathConfig)         automl/ pour gérer leurs outputs                 │
│                                                                            │
│  llm_client.py ─────────► Utilisé par analyse/metier/, pipeline/,          │
│  (OllamaClient)           front/ pour communiquer avec les LLM             │
│                                                                            │
│  io_utils.py ───────────► Utilisé par pipeline/, tests/                    │
│  (chargement données)     pour charger les datasets                        │
│                                                                            │
│  preprocessing.py ──────► Utilisé par pipeline/                            │
│  (split train/test)       pour préparer les données                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. `config.py` - Configuration Centralisée

### Ce qu'il fournit

```python
from src.core.config import settings

# Clés API
settings.openai_api_key          # str | None
settings.huggingface_api_key     # str | None

# Chemins
settings.output_dir              # Path("outputs")
settings.data_dir                # Path("data")

# Méthodes
settings.is_configured("openai") # bool - vérifie si la clé existe
settings.require_api_key("openai") # str - lève ValueError si absente
```

### Où c'est utilisé

| Module | Fichier | Usage |
|--------|---------|-------|
| `core` | `base_path_config.py` | Récupère `settings.output_dir` comme racine par défaut |
| `core` | `llm_client.py` | Récupère les clés API et URLs |
| `front` | `pipeline_streamlit.py` | Vérifie `settings.is_configured("openai")` pour afficher les options |
| `comparison` | `collector.py` | Accède à `settings.output_dir` |

### Exemple concret

```python
# Dans src/front/pipeline_streamlit.py (ligne 44)
from src.core.config import settings

# Vérifier si OpenAI est configuré avant d'afficher les options LLM
if settings.is_configured("openai"):
    st.sidebar.selectbox("Modèle", ["gpt-4o-mini", "gpt-4o"])
```

### Pourquoi c'est centralisé ?

**Problème sans centralisation :**
```python
# ❌ Mauvais - clé API dupliquée partout
api_key = os.getenv("OPENAI_API_KEY")  # Dans chaque fichier
```

**Solution avec `settings` :**
```python
# ✅ Bon - un seul point d'accès
from src.core.config import settings
api_key = settings.openai_api_key
```

**Avantages :**
- Changer la source de config → modifier UN fichier
- Tests : mocker `settings` une seule fois
- Validation centralisée des clés

---

## 2. `base_path_config.py` - Gestion des Chemins

### Ce qu'il fournit

```python
from src.core.base_path_config import BasePathConfig

class BasePathConfig(ABC):
    """Classe abstraite pour gérer les chemins de sortie."""

    MODULE_NAME = "base"  # À surcharger

    def __init__(self, project_name: str, base_dir: Path | None = None):
        self.project_name = project_name
        self.project_dir = base_dir / project_name / self.MODULE_NAME
        self.logs_dir = self.project_dir / "logs"

    # Méthodes concrètes (héritées)
    def log(self, message: str): ...
    def save_json(self, data: dict, path: Path): ...
    def save_metadata(self, metadata: dict): ...

    # Méthodes abstraites (à implémenter)
    @abstractmethod
    def _get_subdirectories(self) -> list[Path]: ...

    @abstractmethod
    def get_all_paths(self) -> dict[str, str]: ...
```

### Qui hérite de cette classe

| Module | Classe | Fichier |
|--------|--------|---------|
| `analyse` | `AnalysePathConfig` | `src/analyse/path_config.py` |
| `feature_engineering` | `FeatureEngineeringPathConfig` | `src/feature_engineering/path_config.py` |
| `automl` | `AutoMLPathConfig` | `src/automl/path_config.py` |

### Exemple concret : `AnalysePathConfig`

```python
# Dans src/analyse/path_config.py
from src.core.base_path_config import BasePathConfig

class AnalysePathConfig(BasePathConfig):
    MODULE_NAME = "analyse"  # → outputs/{projet}/analyse/

    def __init__(self, project_name: str, base_dir=None):
        super().__init__(project_name, base_dir)

        # Définir les sous-dossiers SPÉCIFIQUES à l'analyse
        self.stats_dir = self.project_dir / "stats"
        self.full_dir = self.project_dir / "full"
        self.agent_llm_dir = self.project_dir / "agent_llm"

        self._create_directories()

    def _get_subdirectories(self) -> list[Path]:
        """Retourne les dossiers à créer."""
        return [self.stats_dir, self.agent_llm_dir]

    def get_all_paths(self) -> dict[str, str]:
        """Retourne tous les chemins."""
        return {
            **self.get_base_paths(),
            "stats_dir": str(self.stats_dir),
            "full_dir": str(self.full_dir),
        }

    # Méthodes spécifiques au module
    def save_stats_report(self, report: dict) -> Path:
        return self.save_json(report, self.stats_dir / "report_stats.json")
```

### Structure générée

```
outputs/
└── titanic/                    # project_name
    ├── analyse/                # MODULE_NAME = "analyse"
    │   ├── stats/
    │   ├── full/
    │   ├── agent_llm/
    │   └── logs/
    │
    ├── feature_engineering/    # MODULE_NAME = "feature_engineering"
    │   ├── features/
    │   ├── llmfe/
    │   └── logs/
    │
    └── automl/                 # MODULE_NAME = "automl"
        ├── flaml/
        ├── h2o/
        └── logs/
```

### Pourquoi ce pattern ?

**Problème sans `BasePathConfig` :**
```python
# ❌ Mauvais - chaque module gère ses chemins différemment
output_path = f"outputs/{project}/{module}"  # Duplicaté partout
os.makedirs(output_path, exist_ok=True)      # Duplicaté partout
```

**Solution avec héritage :**
```python
# ✅ Bon - comportement commun dans la classe parente
class MonModulePathConfig(BasePathConfig):
    MODULE_NAME = "mon_module"
    # Tout le reste est hérité !
```

**Avantages :**
- Structure cohérente entre modules
- Logging automatique
- Sauvegarde JSON avec gestion d'erreurs
- Métadonnées standardisées

---

## 3. `llm_client.py` - Client LLM Unifié

### Ce qu'il fournit

```python
from src.core.llm_client import OllamaClient, LLMError, LLMTimeoutError

client = OllamaClient(
    provider="openai",       # ou "ollama"
    model="gpt-4o-mini",
    temperature=0.2,
    format_llm="json",       # Force réponse JSON
)

response = client.chat([
    {"role": "system", "content": "Tu es un expert."},
    {"role": "user", "content": "Analyse ce dataset."},
])
```

### Où c'est utilisé

| Module | Fichier | Usage |
|--------|---------|-------|
| `analyse` | `metier/chatbot_llm.py` | Conversation interactive avec LLM |
| `analyse` | `metier/business_agent.py` | Orchestration clarification métier |
| `pipeline` | `pipeline_all.py` | Lancement de l'agent business |
| `front` | `pipeline_streamlit.py` | Chat LLM dans Streamlit |
| `front` | `components/chat_component.py` | Composant chat |

### Exemple concret

```python
# Dans src/analyse/metier/chatbot_llm.py
from src.core.llm_client import OllamaClient

class BusinessClarificationBot:
    def __init__(self, stats: dict, llm: OllamaClient):
        self.stats = stats
        self.llm = llm  # Client passé en paramètre
        self.messages = []

    def ask_next(self, user_answer: str | None = None) -> str:
        # Ajouter le message user
        if user_answer:
            self.messages.append({"role": "user", "content": user_answer})

        # Appeler le LLM via le client unifié
        response = self.llm.chat(self.messages)

        self.messages.append({"role": "assistant", "content": response})
        return response
```

```python
# Dans src/pipeline/pipeline_all.py (ligne 643)
from src.core.llm_client import OllamaClient

def _run_llm_analysis(self):
    # Créer le client LLM
    llm_client = OllamaClient(
        provider=self.analyse_provider,  # "openai" ou "ollama"
        model=self.analyse_model,
    )

    # L'utiliser pour créer le bot
    bot = BusinessClarificationBot(
        stats=existing_report,
        llm=llm_client,
    )
```

### Pourquoi un client unifié ?

**Problème sans `OllamaClient` :**
```python
# ❌ Mauvais - code différent pour chaque provider
if provider == "openai":
    client = openai.OpenAI(api_key=...)
    response = client.chat.completions.create(...)
elif provider == "ollama":
    response = requests.post("http://localhost:11434/api/chat", ...)
```

**Solution avec `OllamaClient` :**
```python
# ✅ Bon - même interface quel que soit le provider
client = OllamaClient(provider=provider, model=model)
response = client.chat(messages)
```

**Avantages :**
- Ajouter un provider → modifier UN fichier
- Retry automatique avec backoff exponentiel
- Gestion des erreurs standardisée (LLMTimeoutError, etc.)
- Même code pour OpenAI et Ollama

---

## 4. `io_utils.py` - Chargement des Données

### Ce qu'il fournit

```python
from src.core.io_utils import (
    csv_to_dataframe_train_test,  # Charger train/test depuis un dossier
    load_datasets_iris,           # Datasets sklearn
    load_datasets_breast_cancer,
    to_csv,                       # Sauvegarder DataFrame
)

# Charger depuis dossier
train_df, test_df = csv_to_dataframe_train_test("data/raw/titanic")

# Charger dataset sklearn
df = load_datasets_iris()
```

### Où c'est utilisé

| Module | Fichier | Usage |
|--------|---------|-------|
| `pipeline` | `pipeline_autoMl.py` | Charger train/test depuis dossier |
| `tests` | `test_*.py` | Charger datasets sklearn pour tests |

### Exemple concret

```python
# Dans src/pipeline/pipeline_autoMl.py (ligne 48)
from src.core.io_utils import csv_to_dataframe_train_test

def pipeline_create_model(project_name: str, target_col: str, ...):
    data_path = f"data/raw/{project_name}"

    # Charge train.csv et test.csv (si présent)
    df_train, df_test = csv_to_dataframe_train_test(data_path)
    # df_test peut être None si pas de fichier test
```

---

## 5. `preprocessing.py` - Préparation des Données

### Ce qu'il fournit

```python
from src.core.preprocessing import df_to_list, df_to_list_kaggle

# Split interne avec stratification
X_train, X_test, y_train, y_test = df_to_list(
    df,
    target_col="Survived",
    test_size=0.2,
    stratify=True,
)

# Données Kaggle pré-splitées
X_train, X_test, y_train = df_to_list_kaggle(
    df_train,
    df_test,
    target_col="Survived",
)
```

### Où c'est utilisé

| Module | Fichier | Usage |
|--------|---------|-------|
| `pipeline` | `pipeline_autoMl.py` | Préparer données pour AutoML |

### Exemple concret

```python
# Dans src/pipeline/pipeline_autoMl.py (ligne 59)
from src.core.preprocessing import df_to_list_kaggle

def pipeline_create_model(...):
    df_train, df_test = csv_to_dataframe_train_test(data_path)

    # Préparer pour l'entraînement
    X_train, X_test, y_train = df_to_list_kaggle(
        df_train,
        df_test,
        target_col=target_col,
    )

    # X_test a les mêmes colonnes que X_train (alignement auto)
```

---

## Comment Créer un Nouveau Module

### Pattern à Suivre

Si tu veux créer un module `src/mon_module/`, voici le pattern :

#### 1. Créer `path_config.py`

```python
# src/mon_module/path_config.py
from src.core.base_path_config import BasePathConfig
from pathlib import Path

class MonModulePathConfig(BasePathConfig):
    MODULE_NAME = "mon_module"

    def __init__(self, project_name: str, base_dir=None):
        super().__init__(project_name, base_dir)

        # Tes sous-dossiers
        self.results_dir = self.project_dir / "results"
        self.models_dir = self.project_dir / "models"

        self._create_directories()

    def _get_subdirectories(self) -> list[Path]:
        return [self.results_dir, self.models_dir]

    def get_all_paths(self) -> dict[str, str]:
        return {
            **self.get_base_paths(),
            "results_dir": str(self.results_dir),
            "models_dir": str(self.models_dir),
        }

    # Méthodes spécifiques
    def save_results(self, data: dict) -> Path:
        return self.save_json(data, self.results_dir / "results.json")
```

#### 2. Utiliser le PathConfig dans ton module

```python
# src/mon_module/runner.py
from src.mon_module.path_config import MonModulePathConfig

def run(project_name: str, ...):
    paths = MonModulePathConfig(project_name)

    paths.log("Démarrage du module")

    # Ton code...
    results = do_something()

    # Sauvegarder
    paths.save_results(results)
    paths.log("Terminé")
```

#### 3. Si tu utilises un LLM

```python
# src/mon_module/llm_helper.py
from src.core.llm_client import OllamaClient, LLMError

def analyze_with_llm(data: dict, provider: str, model: str) -> dict:
    client = OllamaClient(
        provider=provider,
        model=model,
        format_llm="json",
    )

    try:
        response = client.chat([
            {"role": "system", "content": "Tu es un expert."},
            {"role": "user", "content": f"Analyse: {data}"},
        ])
        return json.loads(response)
    except LLMError as e:
        logger.error(f"Erreur LLM: {e}")
        return {"error": str(e)}
```

---

## Résumé des Imports Clés

```python
# Configuration (TOUJOURS disponible)
from src.core.config import settings

# Gestion des chemins (pour créer un module)
from src.core.base_path_config import BasePathConfig

# Client LLM (pour utiliser OpenAI/Ollama)
from src.core.llm_client import OllamaClient, LLMError, LLMTimeoutError

# I/O données (pour charger/sauver)
from src.core.io_utils import csv_to_dataframe_train_test, to_csv

# Preprocessing (pour split train/test)
from src.core.preprocessing import df_to_list, df_to_list_kaggle
```
