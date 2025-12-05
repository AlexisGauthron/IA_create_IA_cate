# Module Analyse

Profilage statistique du dataset + enrichissement métier via LLM.

📚 **Documentation complète** : [docs/modules/analyse.md](../../docs/modules/analyse.md)

## Fichiers Principaux

| Fichier | Rôle |
|---------|------|
| `analyse.py` | Point d'entrée - orchestre analyse + LLM |
| `path_config.py` | Gestion chemins outputs |
| `statistiques/report.py` | `analyze_dataset_for_fe()` - analyse centrale |
| `statistiques/config.py` | Seuils configurables (FEAnalysisConfig) |
| `correlation/correlation.py` | Pearson, Spearman, MI, MIC, PhiK |
| `metier/chatbot_llm.py` | Dialogue interactif avec LLM |

## Sous-dossiers

| Dossier | Contenu |
|---------|---------|
| `dataset/` | Dataclasses pour structures LLM |
| `statistiques/` | Analyse statistique (targets, features, leakage) |
| `correlation/` | Corrélations avancées |
| `metier/` | Interaction LLM et prompts |
| `helper/` | Utilitaires (JSON, compression) |

## Usage Rapide

```python
from src.analyse.statistiques.report import analyze_dataset_for_fe

report = analyze_dataset_for_fe(df, target_cols="Survived")
llm_payload = report["llm_payload"]
```
