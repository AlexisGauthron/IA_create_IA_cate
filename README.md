# Projet IA creation IA 


## Sujet 


Création d’un pipeline automatisé pour la génération et l’apprentissage de modèle de classification
Le projet vise à concevoir un pipeline intelligent, basé sur un modèle de langage (LLM), capable de créer et d’entraîner automatiquement des modèles de classification à partir d’exemples ou de jeux de données.

L’idée est de permettre à un utilisateur de fournir :

soit quelques extraits ou exemples représentatifs,
soit un dataset complet,
et que le système soit capable de générer un modèle adapté, qu’il s’agisse d’un modèle de classification, de prédiction, ou de tout autre type d’apprentissage supervisé.

Le LLM agit comme un chef d’orchestre, en guidant :

la préparation et l’analyse des données,
la création du modèle le plus approprié,
et son apprentissage automatique.



## Architecture Projet 

```
-Projet/
    | -- src/
    |       | -- analyse_donne/
    |               | -- 
    |
    |       | -- architecture/
    |               | --
    |
    |       | -- optimisations_hyperparametres/
    |               | --
    |
    | -- Test/
    |
    | -- Data/
    |
    | -- Notebook/
    |
    | -- README.md
    | -- Source.ipynb
    | -- pyproject.toml
```


## Architecture technique

https://lucid.app/lucidspark/9022eca2-d866-4ee7-839f-8f6d71c3aae0/edit?page=0_0&invitationId=inv_542a7009-6b65-44b7-98f4-956023edbf70#


## Environnement

### ***Poetry*** pour windows

### ***Conda*** pour Mac

```bash
conda activate Ia_create_ia
```

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli_reference.md) | Paramètres CLI du pipeline |
| [Business Agent](docs/business_agent.md) | Agent de clarification métier LLM |
| [Testing](docs/testing.md) | Guide des tests |

## Quick Start

### Analyse simple

```bash
python tests/integration/test_analyse.py \
    --dataset titanic \
    --target Survived
```

### Avec enrichissement LLM

```bash
python tests/integration/test_analyse.py \
    --dataset titanic \
    --target Survived \
    --with-llm
```

### Pipeline complet

```bash
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --full
```

### App

```bash
streamlit run src/front/interface_streamlit.py
```