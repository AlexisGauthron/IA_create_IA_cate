---
title: "IA Create IA - Pipeline ML intelligent avec Feature Engineering par LLM"
description: "Pipeline end-to-end qui transforme des donnees brutes en modeles entraines grace a un feature engineering pilote par LLM et une orchestration multi-framework AutoML."
tags: ["Machine Learning", "LLM", "AutoML", "Feature Engineering", "Python", "Streamlit"]
date: "2025-11"
duration: "2 mois"
role: "Developpeur ML / Data Scientist"
status: "en_cours"
---

## Description

IA Create IA est un pipeline de Machine Learning complet qui automatise l'ensemble du processus, de l'analyse exploratoire des donnees jusqu'a l'entrainement et la selection du meilleur modele. Le projet se distingue par son approche innovante du feature engineering : un LLM genere des hypotheses de features sous forme de code Python, les evalue via des modeles ML, et itere grace a un systeme de memoire multi-population inspire des algorithmes evolutionnaires.

Le projet repond a un besoin concret : reduire le temps et l'expertise necessaires pour construire un modele ML performant. Plutot que de coder manuellement chaque transformation de features, le pipeline delegue cette tache creative a un LLM qui explore l'espace des features de maniere autonome, tout en conservant un controle humain via une interface Streamlit intuitive.

L'architecture modulaire permet d'utiliser chaque composant independamment (analyse seule, feature engineering seul, AutoML seul) ou de les chainer dans un pipeline complet. Le systeme de cache hierarchique evite les calculs redondants et permet de reprendre une execution interrompue.

## Fonctionnalites principales

- Pipeline 3 etapes configurable : analyse statistique, feature engineering, AutoML
- Feature engineering par LLM (LLMFE) avec memoire multi-population et apprentissage in-context
- Orchestration de 4 frameworks AutoML (FLAML, AutoGluon, H2O, TPOT) avec comparaison automatique
- Detection automatique du type de tache (classification/regression) et des parametres optimaux
- Evaluation multi-metrique ponderee pour la selection de features et de modeles
- Deep Feature Synthesis (DFS) et mode hybride combinant LLMFE et DFS
- Interface Streamlit complete avec suivi en temps reel du pipeline
- Systeme de cache hierarchique et gestionnaire d'experiences pour la reproductibilite
- Support multi-provider LLM : OpenAI (GPT-4o) et Ollama (modeles locaux)
- 3 interfaces : CLI, Streamlit, API Python

## Stack technique

| Categorie | Technologies |
|-----------|-------------|
| Langage | Python 3.11, asyncio |
| ML / Data | scikit-learn, XGBoost, LightGBM, CatBoost, pandas, numpy |
| AutoML | FLAML, AutoGluon, H2O, TPOT |
| Feature Engineering | FeatureTools (DFS), Feature-Engine, LLMFE (custom) |
| LLM | OpenAI (GPT-4o, GPT-3.5-turbo), Ollama (Mistral, Llama3) |
| Interface | Streamlit |
| Qualite | ruff, pytest, type hints complets |
| Gestion | conda, python-dotenv, Conventional Commits |

## Architecture

```
Donnees brutes (CSV)
       |
       v
+------------------------------+
|  STAGE 1 : ANALYSE           |
|  - Stats descriptives         |
|  - Detection du probleme      |
|  - Analyse des correlations   |
|  - Enrichissement LLM metier  |
+------------------------------+
       |
       v
+------------------------------+
|  STAGE 2 : FEATURE ENG.      |
|  - LLMFE (generation par LLM) |
|  - DFS (FeatureTools)         |
|  - Hybride (LLMFE + DFS)     |
|  - Evaluation multi-modele    |
+------------------------------+
       |
       v
+------------------------------+
|  STAGE 3 : AUTOML            |
|  - FLAML (rapide)             |
|  - AutoGluon (stacking)      |
|  - H2O (export MOJO)         |
|  - TPOT (genetique)          |
|  - Comparaison et selection   |
+------------------------------+
       |
       v
   Modele entraine + rapport
```

Chaque stage est modulaire et configurable independamment. Le cache hierarchique persiste les resultats intermediaires (stats, correlations, features, modeles) pour eviter les recalculs. Un gestionnaire d'experiences assure la tracabilite de chaque execution.

## Defis et apprentissages

- Concevoir le systeme LLMFE avec memoire multi-population a necessite de combiner des concepts d'algorithmes evolutionnaires (mutation, crossover, selection) avec du prompt engineering avance pour que le LLM genere du code Python valide et performant
- L'orchestration de 4 frameworks AutoML aux interfaces tres differentes (FLAML synchrone, H2O client-serveur, AutoGluon predictor, TPOT genetique) a demande de creer une couche d'abstraction unifiee gerant les particularites de chacun
- La mise en place du cache hierarchique a impose de definir precisement les cles d'invalidation pour chaque composant, en evitant les recalculs tout en garantissant la coherence des resultats
- L'integration d'un LLM dans une boucle d'evaluation ML a souleve des problemes de robustesse (code genere invalide, timeouts, gestion d'erreurs) resolus par un systeme de retry avec fallback et validation du code avant execution
