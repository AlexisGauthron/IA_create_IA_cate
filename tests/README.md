# Tests

Ce dossier contient les tests du projet **IA_create_IA_cate**.

> **Documentation complète** : Voir [`docs/testing.md`](../docs/testing.md)

---

## Quickstart

```bash
# Activer l'environnement
conda activate Ia_create_ia

# Tous les tests
pytest tests/ -v

# Tests unitaires
pytest tests/unit/ -v

# Tests d'intégration
pytest tests/integration/ -v
```

---

## Structure

```
tests/
├── conftest.py              # Fixtures partagées
├── unit/                    # Tests isolés par composant
│   ├── test_analyse.py
│   ├── test_autogluon.py
│   ├── test_flaml.py
│   ├── test_h2o.py
│   └── test_tpot.py
│
└── integration/             # Tests end-to-end
    ├── test_analyse.py      # Analyse statistique
    ├── test_llmfe.py        # Feature Engineering via LLM
    ├── test_pipeline.py     # Pipeline AutoML complet
    └── test_automl.py
```

---

## Commandes fréquentes

| Commande | Description |
|----------|-------------|
| `python tests/integration/test_analyse.py` | Analyse stats du dataset Titanic |
| `python tests/integration/test_analyse.py --all` | Analyse tous les datasets |
| `python tests/integration/test_analyse.py --with-llm` | Analyse avec LLM interactif |
| `python tests/integration/test_llmfe.py --dry-run` | LLMFE validation config |
| `python tests/integration/test_llmfe.py` | LLMFE feature engineering |
| `python tests/integration/test_pipeline.py` | Pipeline AutoML complet |
| `pytest tests/unit/test_flaml.py -v` | Test unitaire FLAML |

---

## Aide détaillée

Chaque script d'intégration dispose d'une aide intégrée :

```bash
python tests/integration/test_analyse.py --help
python tests/integration/test_llmfe.py --help
python tests/integration/test_pipeline.py --help
```

---

## Voir aussi

- [Documentation complète des tests](../docs/testing.md)
- [README principal](../README.md)
