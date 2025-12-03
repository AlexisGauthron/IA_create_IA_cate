# Agent de Clarification Métier

Documentation de l'agent LLM qui enrichit l'analyse statistique avec des informations métier.

## Vue d'ensemble

L'agent de clarification métier est un composant interactif qui utilise un LLM pour :
1. Poser des questions métier à l'utilisateur
2. Enrichir le payload statistique avec des descriptions business
3. Recommander la métrique d'évaluation la plus adaptée au contexte

## Architecture

```
src/analyse/metier/
├── business_agent.py      # Orchestration de la conversation
├── chatbot_llm.py         # Communication bas niveau avec le LLM
├── prompt_metier.py       # Prompts système
└── parsing_json.py        # Parsing des réponses LLM
```

## Usage

### API Python

```python
from src.analyse.metier.business_agent import run_business_clarification

# Lancer l'agent interactif
enriched_payload = run_business_clarification(
    stats_payload=stats_payload,    # JSON stats du dataset
    provider="openai",              # "openai" ou "ollama"
    model="gpt-4o-mini",            # Modèle LLM
    interactive=True,               # True = input(), False = mode auto
    verbose=True,                   # Affiche les messages
    max_questions=10,               # Max questions avant rapport final
)
```

### CLI

```bash
python tests/integration/test_analyse.py \
    --dataset titanic \
    --target Survived \
    --with-llm \
    --provider openai \
    --model gpt-4o-mini
```

## Commandes utilisateur

Pendant la conversation, l'utilisateur peut utiliser ces commandes :

| Commande | Alternatives | Action |
|----------|--------------|--------|
| `skip` | `passe`, `suivant`, `next` | Passe la question actuelle |
| `done` | `fin`, `stop`, `terminer`, `fini` | Termine et génère le rapport final |

## Champs enrichis

L'agent enrichit le payload avec les champs suivants :

### Context

| Champ | Description |
|-------|-------------|
| `business_description` | Description métier du dataset (2-4 phrases) |
| `final_metric` | Métrique recommandée (`f1`, `accuracy`, `recall`, `roc_auc`, `rmse`) |
| `final_metric_reason` | Justification métier du choix de la métrique |

### Features

| Champ | Description |
|-------|-------------|
| `feature_description` | Description métier de chaque feature |

## Robustesse

L'agent est conçu pour **ne jamais crasher** et **toujours produire un résultat**.

### Gestion des erreurs LLM

| Erreur | Comportement |
|--------|--------------|
| `LLMTimeoutError` | 3 retries avec backoff exponentiel (2s, 4s, 8s) |
| `LLMConnectionError` | 3 retries, puis retour du payload original |
| `LLMRateLimitError` | Délai doublé entre retries (4s, 8s, 16s) |
| Erreur pendant conversation | Tente de forcer le rapport Final |

### Gestion du JSON invalide

```
1. Tentative de parsing direct (json.loads)
   ↓ Échec
2. Extraction regex du bloc JSON
   ↓ Échec
3. Retour du payload original sans enrichissement
```

### Limite de contexte

Pour éviter de dépasser le contexte du LLM :
- Maximum 20 messages dans l'historique (hors system)
- Troncature automatique : garde [system] + [premier message JSON] + [20 derniers]

## Flux de conversation

```
┌─────────────────────────────────────────┐
│ 1. Compacter le payload stats           │
│    (réduction tokens)                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 2. Envoyer le JSON au LLM               │
│    → LLM analyse et pose une question   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 3. Boucle Q&A                           │
│    - LLM pose une question              │
│    - Utilisateur répond (ou skip/done)  │
│    - Répéter jusqu'à Mode: Final        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 4. Fusion des annotations               │
│    stats_payload + réponse LLM          │
└────────────────┬────────────────────────┘
                 │
                 ▼
           Payload enrichi
```

## Format de réponse LLM

### Mode Question

```json
{
  "Mode": "Question",
  "Q": "Le dataset semble concerner la survie de passagers. S'agit-il du Titanic ?"
}
```

### Mode Final

```json
{
  "Mode": "Final",
  "context": {
    "business_description": {
      "value": "Dataset de classification binaire pour prédire la survie...",
      "confidence": 0.9
    },
    "final_metric": {
      "value": "f1",
      "confidence": 0.85
    },
    "final_metric_reason": {
      "value": "Le F1-score est adapté car le dataset est légèrement déséquilibré...",
      "confidence": 0.85
    }
  },
  "features": [
    {
      "name": "Pclass",
      "feature_description": {
        "value": "Classe de voyage du passager (1ère, 2ème, 3ème)",
        "confidence": 0.95
      }
    }
  ]
}
```

## Détection du Mode Final

L'agent utilise une détection robuste :

1. **Parsing JSON** : Vérifie `data["Mode"].lower() == "final"`
2. **Fallback regex** : Cherche `"Mode": "Final"` (insensible à la casse)

## Métriques suggérées vs finales

Le pipeline utilise deux niveaux de métriques :

| Source | Champ | Description |
|--------|-------|-------------|
| Statistique | `suggested_metric` | Basée sur des seuils (imbalance ratio, etc.) |
| LLM | `final_metric` | Validée/modifiée par le LLM selon le contexte métier |

**Priorité** : `final_metric` (LLM) > `suggested_metric` (seuils)

## Exceptions personnalisées

Définies dans `src/core/llm_client.py` :

```python
class LLMError(Exception):
    """Classe de base pour les erreurs LLM."""

class LLMTimeoutError(LLMError):
    """Le LLM n'a pas répondu dans le temps imparti."""

class LLMConnectionError(LLMError):
    """Erreur de connexion au LLM."""

class LLMRateLimitError(LLMError):
    """Rate limit atteint."""
```

## Configuration

### Constantes modifiables

| Fichier | Constante | Défaut | Description |
|---------|-----------|--------|-------------|
| `llm_client.py` | `MAX_RETRIES` | 3 | Nombre de tentatives |
| `llm_client.py` | `RETRY_BASE_DELAY` | 2s | Délai de base (backoff) |
| `chatbot_llm.py` | `MAX_CONVERSATION_MESSAGES` | 20 | Limite messages historique |
| `business_agent.py` | `max_questions` | 10 | Max questions par session |

## Exemples

### Analyse simple avec LLM

```bash
python tests/integration/test_analyse.py \
    --dataset titanic \
    --target Survived \
    --with-llm
```

### Avec Ollama local

```bash
python tests/integration/test_analyse.py \
    --dataset titanic \
    --target Survived \
    --with-llm \
    --provider ollama \
    --model llama3
```

### Mode non-interactif

```python
# Le LLM fait ses hypothèses sans poser de questions
enriched = run_business_clarification(
    stats_payload=payload,
    interactive=False,  # Pas d'input()
)
```
