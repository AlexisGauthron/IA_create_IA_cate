# Module Front

Interfaces Streamlit : Pipeline ML complet + Classification few-shot.

📚 **Documentation complète** : [docs/modules/front.md](../../docs/modules/front.md)

## Interfaces

| Fichier | Description | Commande |
|---------|-------------|----------|
| `pipeline_streamlit.py` | Pipeline ML complet (7 étapes) | `streamlit run src/front/pipeline_streamlit.py` |
| `interface_streamlit.py` | Classification few-shot | `streamlit run src/front/interface_streamlit.py` |

## Pipeline ML (7 Étapes)

1. **Upload** : Chargement CSV (train + test)
2. **Configuration** : Cible, type de tâche, config LLM
3. **Analyse** : Stats descriptives + corrélations
4. **Agent métier** : Chat pour clarifier le contexte
5. **Feature Engineering** : LLMFE avec suivi temps réel
6. **AutoML** : Entraînement multi-frameworks
7. **Résultats** : Synthèse finale

## Fichiers Utilitaires

| Fichier | Rôle |
|---------|------|
| `upload_fichier.py` | Lecture/fusion CSV |
| `fe_runner_async.py` | Exécution LLMFE async |
| `fe_progress_monitor.py` | Suivi progression LLMFE |
| `css.py` | Thème CSS personnalisé |
| `ui_helper.py` | Helpers Streamlit |

## Composants (`components/`)

| Composant | Usage |
|-----------|-------|
| `chat_component.py` | Chat réutilisable (agent métier) |
| `llmfe_visualizer.py` | Dashboard résultats LLMFE |
| `results_component.py` | Affichage résultats |

## Lancement

```bash
# Activer l'environnement
conda activate Ia_create_ia

# Pipeline ML complet
streamlit run src/front/pipeline_streamlit.py

# Few-shot classification
streamlit run src/front/interface_streamlit.py
```
