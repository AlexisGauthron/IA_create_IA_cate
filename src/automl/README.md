# Module AutoML

Orchestration de 4 frameworks AutoML : FLAML, AutoGluon, TPOT, H2O.

📚 **Documentation complète** : [docs/modules/automl.md](../../docs/modules/automl.md)

## Fichiers Principaux

| Fichier | Rôle |
|---------|------|
| `runner.py` | Orchestrateur (`AutoMLRunner`) |
| `path_config.py` | Gestion centralisée des chemins |

## Wrappers (supervised/)

| Fichier | Framework |
|---------|-----------|
| `flaml_wrapper.py` | FLAML |
| `autogluon_wrapper.py` | AutoGluon |
| `h2o_wrapper.py` | H2O AutoML |
| `tpot_wrapper.py` | TPOT |

## Usage Rapide

```python
from src.automl.runner import AutoMLRunner

runner = AutoMLRunner(output_dir, X_train, X_test, y_train, y_test)
runner.use_all(model=["flaml", "autogluon", "h2o", "tpot"])
runner.compare_all_predict(model=["flaml", "autogluon", "h2o", "tpot"])

print(f"FLAML: {runner.score_flaml}")
print(f"H2O: {runner.score_h2o}")
```

## Comparaison Frameworks

| Framework | Vitesse | Export |
|-----------|---------|--------|
| FLAML | ⚡⚡⚡ | joblib |
| AutoGluon | ⚡⚡ | predictor |
| H2O | ⚡⚡ | MOJO |
| TPOT | ⚡ | sklearn |
