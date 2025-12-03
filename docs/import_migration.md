# Guide de migration des imports

Ce fichier liste les anciens imports et leurs équivalents dans la nouvelle architecture.

## Mapping

| Ancien import | Nouvel import |
|---------------|---------------|
| `src.Data.load_datasets` | `src.data.loader` |
| `src.analyse.metier` | `src.analyse.business` |
| `src.analyse.statistiques` | `src.analyse.stats` |
| `src.autoML_nonsupervise` | `src.automl.unsupervised` |
| `src.autoML_supervise` | `src.automl.supervised` |
| `src.autoML_supervise.all_autoML` | `src.automl.runner` |
| `src.features_engineering` | `src.feature_engineering` |
| `src.features_engineering.LLM.analyse_fe` | `src.feature_engineering.declarative` |
| `src.features_engineering.LLM.code_fe` | `src.feature_engineering.declarative` |
| `src.features_engineering.LLM.transcriptions_fe` | `src.feature_engineering.declarative` |
| `src.features_engineering.lib_existante` | `src.feature_engineering.libs` |
| `src.features_engineering.transformation_fe` | `src.feature_engineering.transforms` |
| `src.fonctions.clean_label` | `src.core.text_cleaning` |
| `src.fonctions.csv` | `src.core.io_utils` |
| `src.fonctions.format_entrainement` | `src.core.preprocessing` |
| `src.helper.ddataframe` | `src.core.dataframe_utils` |
| `src.helper.ollama_llm` | `src.core.llm_client` |

## Exemple de migration

```python
# Avant
from src.fonctions.csv import to_csv
from src.autoML_supervise.all_autoML import all_autoML
from src.features_engineering.LLM.analyse_fe.pipeline import LLMFeatureEngineeringPipeline

# Après
from src.core.io_utils import to_csv
from src.automl.runner import all_autoML
from src.feature_engineering.declarative.planner import LLMFeatureEngineeringPipeline
```

## Script de remplacement automatique

Pour mettre à jour vos imports automatiquement, vous pouvez utiliser sed:

```bash
# Exemple pour un fichier
sed -i '' 's/src.fonctions.csv/src.core.io_utils/g' votre_fichier.py
```
