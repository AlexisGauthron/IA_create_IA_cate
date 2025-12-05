# Module Core

> Module fondamental fournissant la configuration, les utilitaires I/O, la gestion des chemins et l'intégration LLM pour l'ensemble du projet.

## Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Fichiers du Module](#fichiers-du-module)
  - [config.py](#configpy---configuration-centralisée)
  - [base_path_config.py](#base_path_configpy---gestion-des-chemins)
  - [llm_client.py](#llm_clientpy---client-llm-unifié)
  - [io_utils.py](#io_utilspy---entrées-sorties)
  - [preprocessing.py](#preprocessingpy---prétraitement)
  - [dataframe_utils.py](#dataframe_utilspy---utilitaires-dataframe)
  - [text_cleaning.py](#text_cleaningpy---nettoyage-texte)
- [Diagramme de Dépendances](#diagramme-de-dépendances)
- [Patterns de Conception](#patterns-de-conception)
- [Exemples d'Utilisation](#exemples-dutilisation)
- [Gestion des Erreurs](#gestion-des-erreurs)

---

## Vue d'Ensemble

Le module `core` est la **couche fondamentale** de l'application. Il fournit :

| Responsabilité | Fichier | Description |
|----------------|---------|-------------|
| **Configuration** | `config.py` | Clés API, chemins, paramètres globaux |
| **Gestion des chemins** | `base_path_config.py` | Classe abstraite pour structure des outputs |
| **Intégration LLM** | `llm_client.py` | Client unifié Ollama/OpenAI |
| **I/O Données** | `io_utils.py` | Chargement/sauvegarde CSV, datasets sklearn |
| **Prétraitement** | `preprocessing.py` | Split train/test avec stratification |
| **Utilitaires DataFrame** | `dataframe_utils.py` | Manipulation pandas |
| **Nettoyage texte** | `text_cleaning.py` | Standardisation labels |

**Tous les autres modules dépendent de `core`** pour leurs besoins fondamentaux.

---

## Architecture

```
src/core/
├── __init__.py              # Package marker
├── config.py                # Configuration centralisée (Settings singleton)
├── base_path_config.py      # Classe abstraite pour gestion des chemins
├── llm_client.py            # Client LLM unifié (Ollama + OpenAI)
├── io_utils.py              # Chargement/sauvegarde données
├── preprocessing.py         # Split train/test
├── dataframe_utils.py       # Utilitaires DataFrame
└── text_cleaning.py         # Nettoyage de labels texte
```

### Flux de Données

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION                             │
│                         (config.py)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ API Keys    │  │ Paths       │  │ Defaults                │  │
│  │ - OpenAI    │  │ - data_dir  │  │ - default_provider      │  │
│  │ - HuggingFace│ │ - output_dir│  │ - default_model         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  LLM Client   │    │  Path Config  │    │  I/O Utils    │
│ (llm_client)  │    │ (base_path)   │    │ (io_utils)    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODULES CONSOMMATEURS                        │
│      analyse/  │  feature_engineering/  │  automl/  │  front/   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fichiers du Module

### `config.py` - Configuration Centralisée

#### Objectif
Gérer toutes les configurations de l'application via un singleton `Settings`.

#### Classe `Settings`

```python
@dataclass
class Settings:
    """Configuration centralisée de l'application."""

    # Clés API
    openai_api_key: str | None      # Depuis OPENAI_API_KEY
    huggingface_api_key: str | None # Depuis HUGGINGFACE_API_KEY

    # URLs
    ollama_base_url: str = "http://localhost:11434/api/chat"

    # Valeurs par défaut
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"

    # Chemins
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    models_dir: Path = Path("Modeles")
```

#### Méthodes

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `get_api_key()` | `(provider: str) -> str \| None` | Récupère une clé API par provider |
| `require_api_key()` | `(provider: str) -> str` | Récupère ou lève `ValueError` |
| `is_configured()` | `(provider: str) -> bool` | Vérifie si un provider est configuré |
| `to_dict()` | `() -> dict` | Retourne config (sans clés sensibles) |

#### Fonctions Utilitaires

```python
# Raccourcis
get_openai_key() -> str           # Clé OpenAI
get_huggingface_key() -> str      # Clé HuggingFace
is_openai_configured() -> bool    # OpenAI disponible ?
is_ollama_available() -> bool     # Serveur Ollama accessible ?
```

#### Instance Singleton

```python
settings = Settings()  # Instance globale accessible partout
```

#### Exemple d'Utilisation

```python
from src.core.config import settings

# Accès direct
api_key = settings.openai_api_key
output = settings.output_dir

# Avec validation
try:
    api_key = settings.require_api_key("openai")
except ValueError:
    print("Clé OpenAI non configurée")

# Vérification
if settings.is_configured("openai"):
    # Utiliser OpenAI
    pass
```

---

### `base_path_config.py` - Gestion des Chemins

#### Objectif
Fournir une classe abstraite pour la gestion cohérente des chemins de sortie dans tous les modules.

#### Classe `BasePathConfig` (ABC)

```python
class BasePathConfig(ABC):
    """Classe abstraite de base pour la gestion des chemins."""

    MODULE_NAME: str = "base"  # À surcharger
```

#### Constructeur

```python
def __init__(
    self,
    project_name: str,           # Nom du projet (ex: "titanic")
    base_dir: str | Path | None  # Racine custom (défaut: settings.output_dir)
) -> None
```

#### Attributs d'Instance

| Attribut | Type | Description |
|----------|------|-------------|
| `project_name` | str | Nom du projet |
| `timestamp` | str | Horodatage (YYYYmmdd_HHMMSS) |
| `project_dir` | Path | `{base_dir}/{project_name}/{MODULE_NAME}/` |
| `logs_dir` | Path | `{project_dir}/logs/` |

#### Méthodes Abstraites (à implémenter)

```python
@abstractmethod
def _get_subdirectories(self) -> list[Path]:
    """Liste des sous-répertoires à créer."""
    pass

@abstractmethod
def get_all_paths(self) -> dict[str, str]:
    """Dictionnaire de tous les chemins configurés."""
    pass
```

#### Méthodes Concrètes

| Méthode | Description |
|---------|-------------|
| `log(message, level)` | Écrit dans le fichier log |
| `save_json(data, path)` | Sauvegarde dict en JSON |
| `save_metadata(metadata)` | Sauvegarde métadonnées projet |
| `get_base_paths()` | Retourne chemins de base |
| `from_existing(project_dir)` | Crée instance depuis dossier existant |

#### Structure de Répertoires Générée

```
outputs/
└── {project_name}/
    └── {MODULE_NAME}/
        ├── logs/
        │   └── {MODULE_NAME}.log
        ├── metadata.json
        └── {subdirectories}/
```

#### Exemple d'Implémentation

```python
from src.core.base_path_config import BasePathConfig

class AnalysePathConfig(BasePathConfig):
    MODULE_NAME = "analyse"

    def __init__(self, project_name: str, base_dir=None):
        super().__init__(project_name, base_dir)
        # Définir sous-répertoires spécifiques
        self.stats_dir = self.project_dir / "stats"
        self.full_dir = self.project_dir / "full"
        self._create_directories()

    def _get_subdirectories(self) -> list[Path]:
        return [self.stats_dir, self.full_dir]

    def get_all_paths(self) -> dict[str, str]:
        return {
            **self.get_base_paths(),
            "stats_dir": str(self.stats_dir),
            "full_dir": str(self.full_dir),
        }

# Utilisation
paths = AnalysePathConfig("titanic")
paths.log("Analyse démarrée")
paths.save_metadata({"n_rows": 891, "n_cols": 12})
```

---

### `llm_client.py` - Client LLM Unifié

#### Objectif
Fournir une interface unifiée pour communiquer avec Ollama (local) et OpenAI (cloud).

#### Exceptions Personnalisées

```python
class LLMError(Exception):
    """Erreur de base pour le LLM."""

class LLMTimeoutError(LLMError):
    """Le LLM n'a pas répondu à temps."""

class LLMConnectionError(LLMError):
    """Erreur de connexion au serveur LLM."""

class LLMRateLimitError(LLMError):
    """Limite de requêtes dépassée."""
```

#### Classe `OllamaClient`

```python
class OllamaClient:
    """Client LLM unifié pour Ollama et OpenAI."""

    def __init__(
        self,
        model: str | None = None,           # Modèle (défaut: settings.default_model)
        *,
        provider: Literal["ollama", "openai"] | None = None,
        # Ollama
        base_url: str | None = None,         # URL serveur Ollama
        # OpenAI
        openai_api_key: str | None = None,   # Clé API
        openai_base_url: str | None = None,  # URL custom
        # Commun
        temperature: float = 0.2,            # Température (0-1)
        max_tokens: int = 8024,              # Tokens max
        format_llm: str | None = None,       # "json" pour JSON mode
    ) -> None
```

#### Méthode Principale

```python
def chat(self, messages: list[dict[str, str]]) -> str:
    """
    Envoie des messages et retourne la réponse.

    Args:
        messages: Liste de messages au format OpenAI
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "..."}]

    Returns:
        Contenu de la réponse du LLM
    """
```

#### Stratégie de Retry

| Paramètre | Valeur |
|-----------|--------|
| Max tentatives | 3 |
| Délai base | 2 secondes |
| Backoff | Exponentiel (2s, 4s, 8s) |
| Rate limit | Délai doublé (4s, 8s, 16s) |
| Timeout HTTP | 120 secondes |

#### Exemple d'Utilisation

```python
from src.core.llm_client import OllamaClient, LLMError

# Client OpenAI
client = OllamaClient(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.2
)

# Client Ollama (local)
client_local = OllamaClient(
    provider="ollama",
    model="mistral",
    base_url="http://localhost:11434/api/chat"
)

# Envoi de messages
try:
    response = client.chat([
        {"role": "system", "content": "Tu es un expert en data science."},
        {"role": "user", "content": "Explique le Random Forest."}
    ])
    print(response)
except LLMTimeoutError:
    print("Timeout - le LLM n'a pas répondu")
except LLMRateLimitError:
    print("Rate limit atteint - attendre avant de réessayer")
except LLMError as e:
    print(f"Erreur LLM: {e}")
```

#### Mode JSON

```python
# Forcer une réponse JSON valide
client = OllamaClient(
    provider="openai",
    model="gpt-4o-mini",
    format_llm="json"  # Active le JSON mode
)

response = client.chat([
    {"role": "user", "content": "Retourne un JSON avec name et age."}
])
# Réponse garantie d'être un JSON valide
import json
data = json.loads(response)
```

---

### `io_utils.py` - Entrées/Sorties

#### Objectif
Fournir des utilitaires pour charger et sauvegarder des données.

#### Datasets Sklearn

```python
# Chargement direct en DataFrame
load_datasets_breast_cancer() -> pd.DataFrame  # 569 lignes, 31 colonnes
load_datasets_iris() -> pd.DataFrame           # 150 lignes, 5 colonnes
load_datasets_wine() -> pd.DataFrame           # 178 lignes, 14 colonnes
load_datasets_digits() -> pd.DataFrame         # 1797 lignes, 65 colonnes
```

#### Chargement CSV Train/Test

```python
def csv_to_dataframe_train_test(
    nom_dossier: str | Path,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    train_pattern: str = "train",
    test_pattern: str = "test",
    **read_csv_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Charge train.csv et test.csv depuis un dossier.

    Args:
        nom_dossier: Chemin du dossier
        sep: Séparateur CSV
        encoding: Encodage
        train_pattern: Pattern pour fichier train (requis)
        test_pattern: Pattern pour fichier test (optionnel)

    Returns:
        (train_df, test_df ou None)

    Raises:
        FileNotFoundError: Dossier ou fichier train introuvable
        ValueError: Plusieurs fichiers correspondent au pattern
    """
```

#### Sauvegarde CSV

```python
def to_csv(
    df: pd.DataFrame,
    nom_fichier: str,
    nom_dossier: str = "Data",
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    index: bool = False,
    mode: str = "w",       # "w" écrase, "x" erreur si existe, "a" append
    header: bool = True,
    na_rep: str = "",
    quoting: int = csv.QUOTE_MINIMAL,
) -> Path:
    """Sauvegarde DataFrame en CSV, crée le dossier si nécessaire."""
```

#### Lecture CSV

```python
def to_dataframe(
    nom_fichier: str,
    nom_dossier: str = "Data",
    index_col: str | None = None,
    parse_dates: list | None = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Charge un fichier CSV en DataFrame."""
```

#### Exemple d'Utilisation

```python
from src.core.io_utils import (
    csv_to_dataframe_train_test,
    to_csv,
    load_datasets_iris
)

# Charger dataset sklearn
df_iris = load_datasets_iris()

# Charger depuis dossier
train_df, test_df = csv_to_dataframe_train_test(
    "data/raw/titanic",
    sep=","
)

# Sauvegarder
to_csv(train_df, "processed_train", "outputs/titanic")
```

---

### `preprocessing.py` - Prétraitement

#### Objectif
Fournir des fonctions pour le split train/test avec stratification.

#### Split avec Stratification

```python
def df_to_list(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare un DataFrame en train/test avec stratification optionnelle.

    Args:
        df: DataFrame complet (features + target)
        target_col: Nom de la colonne cible
        test_size: Proportion pour le test (0 < x < 1)
        random_state: Graine aléatoire
        stratify: Activer stratification (classification)

    Returns:
        (X_train, X_test, y_train, y_test)

    Note:
        Stratification activée seulement si:
        - stratify=True
        - Plus d'une classe
        - Pas de NaN dans target
        - Chaque classe a >= 2 échantillons
    """
```

#### Split Style Kaggle

```python
def df_to_list_kaggle(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "target",
    align_columns: bool = True,
    fill_missing: float | int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Traite des données Kaggle pré-splitées.

    Args:
        df_train: DataFrame train avec target
        df_test: DataFrame test sans target
        target_col: Colonne cible
        align_columns: Aligner colonnes test sur train
        fill_missing: Valeur pour colonnes manquantes

    Returns:
        (X_train, X_test, y_train)
        Note: Pas de y_test (non disponible sur Kaggle)
    """
```

#### Exemple d'Utilisation

```python
from src.core.preprocessing import df_to_list, df_to_list_kaggle
import pandas as pd

# Split standard
df = pd.read_csv("titanic.csv")
X_train, X_test, y_train, y_test = df_to_list(
    df,
    target_col="Survived",
    test_size=0.2,
    stratify=True
)

# Style Kaggle
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
X_train, X_test, y_train = df_to_list_kaggle(
    train_df, test_df,
    target_col="Survived"
)
```

---

### `dataframe_utils.py` - Utilitaires DataFrame

#### Fonctions

```python
def get_unique_columns_dataframe(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    """
    Récupère les colonnes uniques à df2 par rapport à df1.
    Préserve toujours la colonne target.
    """

def drop_columns(
    df: pd.DataFrame,
    columns_to_drop: list
) -> pd.DataFrame:
    """
    Supprime des colonnes (ignore si inexistantes).
    Retourne une copie.
    """

def count_features(df: pd.DataFrame) -> int:
    """Compte le nombre de colonnes."""
```

---

### `text_cleaning.py` - Nettoyage Texte

#### Objectif
Nettoyer et standardiser les labels textuels.

#### Constante

```python
MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--"}
```

#### Fonction

```python
def _clean_labels(seq) -> list[str]:
    """
    Nettoie et déduplique une séquence de labels.

    Processing:
    1. Convertit en Series
    2. Supprime NaN
    3. Convertit en string et strip
    4. Supprime MISSING_TOKENS (case-insensitive)
    5. Déduplique en préservant l'ordre

    Example:
        ["", "Cat", "cat", "nan", "Dog", "DOG"]
        → ["Cat", "Dog"]
    """
```

---

## Diagramme de Dépendances

```
                    ┌─────────────────┐
                    │    config.py    │
                    │   (Settings)    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ base_path_      │ │  llm_client.py  │ │   (externe)     │
│ config.py       │ │  (OllamaClient) │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              MODULES HÉRITANT DE BasePathConfig         │
│  AnalysePathConfig │ LLMFEPathConfig │ AutoMLPathConfig │
└─────────────────────────────────────────────────────────┘

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  io_utils.py    │ │ preprocessing.py│ │ dataframe_utils │
│ (indépendant)   │ │ (indépendant)   │ │ (indépendant)   │
└─────────────────┘ └─────────────────┘ └─────────────────┘

┌─────────────────┐
│ text_cleaning.py│
│ (indépendant)   │
└─────────────────┘
```

**Légende:**
- `config.py` : Point central, aucune dépendance interne
- `base_path_config.py` : Dépend de `config.py` (lazy import)
- `llm_client.py` : Dépend de `config.py` (lazy import)
- Autres fichiers : Indépendants (pandas, sklearn uniquement)

---

## Patterns de Conception

### 1. Singleton Pattern
```python
# config.py
settings = Settings()  # Instance unique globale

# Utilisation partout
from src.core.config import settings
api_key = settings.openai_api_key
```

### 2. Abstract Base Class (Template Method)
```python
# base_path_config.py
class BasePathConfig(ABC):
    def __init__(self, ...):
        # Setup commun
        self._create_directories()  # Appelle méthodes abstraites

    @abstractmethod
    def _get_subdirectories(self): pass  # À implémenter

    @abstractmethod
    def get_all_paths(self): pass  # À implémenter
```

### 3. Strategy Pattern
```python
# llm_client.py
class OllamaClient:
    def chat(self, messages):
        if self.provider == "ollama":
            return self._chat_ollama(messages)
        else:
            return self._chat_openai(messages)
```

### 4. Lazy Import (Anti-Circular Dependencies)
```python
# base_path_config.py
def save_json(self, data, path):
    from src.core.config import settings  # Import tardif
    # ...
```

### 5. Retry with Exponential Backoff
```python
# llm_client.py
for attempt in range(MAX_RETRIES):
    try:
        return self._make_request()
    except Exception:
        delay = RETRY_BASE_DELAY * (2 ** attempt)  # 2s, 4s, 8s
        time.sleep(delay)
```

---

## Exemples d'Utilisation

### Configuration Complète

```python
from src.core.config import settings, is_openai_configured, is_ollama_available

# Vérifier disponibilité
if is_openai_configured():
    print(f"OpenAI configuré avec modèle: {settings.default_model}")

if is_ollama_available():
    print("Serveur Ollama accessible")

# Chemins
print(f"Données: {settings.data_dir}")
print(f"Outputs: {settings.output_dir}")
```

### Pipeline de Chargement Données

```python
from src.core.io_utils import csv_to_dataframe_train_test
from src.core.preprocessing import df_to_list

# 1. Charger depuis dossier
train_df, test_df = csv_to_dataframe_train_test("data/raw/titanic")

# 2. Si pas de test, faire un split
if test_df is None:
    X_train, X_test, y_train, y_test = df_to_list(
        train_df,
        target_col="Survived",
        test_size=0.2,
        stratify=True
    )
else:
    # Utiliser le split Kaggle
    from src.core.preprocessing import df_to_list_kaggle
    X_train, X_test, y_train = df_to_list_kaggle(
        train_df, test_df,
        target_col="Survived"
    )
```

### Conversation LLM

```python
from src.core.llm_client import OllamaClient, LLMError

def analyze_with_llm(data_description: str) -> dict:
    client = OllamaClient(
        provider="openai",
        model="gpt-4o-mini",
        format_llm="json"
    )

    try:
        response = client.chat([
            {
                "role": "system",
                "content": "Tu es un expert data scientist. Réponds en JSON."
            },
            {
                "role": "user",
                "content": f"Analyse ce dataset:\n{data_description}"
            }
        ])
        return json.loads(response)
    except LLMError as e:
        return {"error": str(e)}
```

### Création d'un Module avec PathConfig

```python
from src.core.base_path_config import BasePathConfig
from pathlib import Path

class MonModulePathConfig(BasePathConfig):
    MODULE_NAME = "mon_module"

    def __init__(self, project_name: str, base_dir=None):
        super().__init__(project_name, base_dir)
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

# Utilisation
paths = MonModulePathConfig("titanic")
paths.log("Démarrage du module")
paths.save_json({"status": "ok"}, paths.results_dir / "status.json")
```

---

## Gestion des Erreurs

| Scénario | Exception | Module |
|----------|-----------|--------|
| Clé API manquante | `ValueError` | config.py |
| Dossier introuvable | `FileNotFoundError` | io_utils.py |
| Plusieurs fichiers match | `ValueError` | io_utils.py |
| Colonne target absente | `KeyError` | preprocessing.py |
| Timeout LLM | `LLMTimeoutError` | llm_client.py |
| Connexion LLM échouée | `LLMConnectionError` | llm_client.py |
| Rate limit dépassé | `LLMRateLimitError` | llm_client.py |
| Réponse LLM vide | `LLMError` | llm_client.py |

### Bonnes Pratiques

```python
from src.core.config import settings
from src.core.llm_client import OllamaClient, LLMError, LLMTimeoutError

# 1. Toujours vérifier la configuration
if not settings.is_configured("openai"):
    raise RuntimeError("OpenAI non configuré. Définir OPENAI_API_KEY.")

# 2. Gérer les erreurs LLM
try:
    client = OllamaClient(provider="openai")
    response = client.chat(messages)
except LLMTimeoutError:
    # Retry ou fallback vers Ollama local
    client = OllamaClient(provider="ollama", model="mistral")
    response = client.chat(messages)
except LLMError as e:
    logger.error(f"Erreur LLM: {e}")
    response = None
```

---

## Dépendances Externes

| Package | Version | Usage |
|---------|---------|-------|
| `pandas` | >= 2.0 | Manipulation données |
| `scikit-learn` | >= 1.3 | Split train/test, datasets |
| `openai` | >= 1.0 | Client API OpenAI |
| `requests` | >= 2.28 | HTTP pour Ollama |
| `python-dotenv` | >= 1.0 | Chargement .env |

---

## Variables d'Environnement

Fichier `.env` à la racine du projet :

```env
# API Keys
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...

# Optionnel
OLLAMA_BASE_URL=http://localhost:11434/api/chat
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
```

---

## Statistiques du Module

| Fichier | Lignes | Classes | Fonctions |
|---------|--------|---------|-----------|
| `config.py` | 163 | 1 | 5 |
| `base_path_config.py` | 235 | 1 | 9 |
| `llm_client.py` | 299 | 5 | 3 |
| `io_utils.py` | 198 | 0 | 8 |
| `preprocessing.py` | 113 | 0 | 2 |
| `dataframe_utils.py` | 47 | 0 | 3 |
| `text_cleaning.py` | 28 | 0 | 1 |
| **Total** | **~1,083** | **7** | **31** |
