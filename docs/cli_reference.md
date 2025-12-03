# CLI Reference - Pipeline ML

Documentation complète des paramètres CLI pour le pipeline ML.

## Usage

```bash
python tests/integration/test_pipeline_all.py [OPTIONS]
```

---

## 1. Paramètres Principaux

### Dataset & Projet

| Option | Type | Requis | Défaut | Description |
|--------|------|--------|--------|-------------|
| `--dataset` | str | **OUI** | - | Nom du dataset dans `data/raw/` |
| `--target` | str | **OUI** | - | Colonne cible à prédire |
| `--project` | str | non | (=dataset) | Nom du projet pour les outputs |
| `--output-dir` | str | non | `outputs` | Dossier racine des résultats |

### Mode d'exécution

| Option | Type | Description |
|--------|------|-------------|
| `--analyse-only` | flag | Analyse statistique uniquement |
| `--no-automl` | flag | Analyse + Feature Engineering (sans AutoML) |
| `--full` | flag | Pipeline complet (Analyse + FE + AutoML) |
| `--force-analyse` | flag | Force la regénération de l'analyse même si le JSON existe |
| `--unit-tests` | flag | Lance les tests unitaires |

**Note sur `--force-analyse`** : Par défaut, si une analyse existe déjà pour le projet (fichier `outputs/{project}/analyse/stats/report_stats.json`), elle sera réutilisée. Utilisez `--force-analyse` pour la regénérer.

---

## 2. Analyse des Corrélations

L'analyse des corrélations permet de mesurer la relation entre chaque feature et la cible. Elle est **désactivée par défaut** pour des raisons de performance.

### Options

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--with-correlations` | flag | off | Active l'analyse des corrélations |
| `--correlation-methods` | str | `pearson,spearman,kendall,mutual_info` | Méthodes à utiliser (séparées par virgule) |

### Méthodes disponibles

| Méthode | Description | Dépendance |
|---------|-------------|------------|
| `pearson` | Corrélation linéaire | sklearn (inclus) |
| `spearman` | Corrélation de rang (monotone) | sklearn (inclus) |
| `kendall` | Corrélation de rang (concordance) | sklearn (inclus) |
| `mutual_info` | Information mutuelle (non-linéaire) | sklearn (inclus) |
| `mic` | Maximal Information Coefficient | `minepy` (optionnel) |
| `phik` | Corrélation robuste catégoriel/numérique | `phik` (optionnel) |

### Résultats

Les corrélations sont sauvegardées dans le fichier JSON d'analyse :

```
outputs/{projet}/analyse/stats/report_stats.json
```

Structure du bloc `correlations` :

```json
{
  "correlations": {
    "target": "Survived",
    "task": "classification",
    "methods_used": ["pearson", "spearman", "kendall", "mutual_info"],
    "classical": [
      {"feature": "Age", "pearson": 0.12, "spearman": 0.15, "kendall": 0.10}
    ],
    "mutual_info": [
      {"feature": "Age", "mutual_info": 0.08}
    ],
    "combined_scores": [
      {"feature": "Sex", "combined_score": 0.81, "pearson": 0.54, ...}
    ],
    "top_10_features": [
      {"feature": "Sex", "combined_score": 0.8066},
      {"feature": "Pclass", "combined_score": 0.4827}
    ]
  }
}
```

### Exemples

```bash
# Analyse avec corrélations (méthodes par défaut)
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only \
    --with-correlations

# Avec toutes les méthodes (si minepy et phik installés)
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only \
    --with-correlations \
    --correlation-methods "pearson,spearman,kendall,mutual_info,mic,phik"

# Méthodes rapides uniquement
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only \
    --with-correlations \
    --correlation-methods "pearson,mutual_info"
```

### Performance

| Dataset | Sans corrélations | Avec corrélations (défaut) | Avec MIC+PhiK |
|---------|-------------------|---------------------------|---------------|
| 1000 lignes × 30 features | ~2s | ~6s | ~45s |
| 10000 lignes × 50 features | ~5s | ~20s | ~5min |

**Recommandation** : Pour les gros datasets (>10k lignes), évitez `mic` et `phik`.

---

## 3. Configuration LLM

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--with-llm` | flag | off | Active l'analyse métier LLM |
| `--analyse-provider` | str | `openai` | Provider LLM (openai, ollama) |
| `--analyse-model` | str | `gpt-4o-mini` | Modèle pour analyse métier |
| `--llmfe-model` | str | `gpt-3.5-turbo` | Modèle pour Feature Engineering |

### Agent de clarification métier

Quand `--with-llm` est activé, un agent interactif pose des questions pour enrichir l'analyse avec des informations métier :
- **business_description** : description métier du dataset
- **final_metric** : métrique d'évaluation recommandée par le LLM
- **final_metric_reason** : justification métier du choix de la métrique
- **feature_description** : description métier de chaque feature

#### Commandes disponibles pendant la conversation

| Commande | Action |
|----------|--------|
| `skip`, `passe`, `suivant` | Passer la question actuelle |
| `done`, `fin`, `stop`, `terminer` | Terminer et générer le rapport final |

#### Robustesse de l'agent

L'agent est conçu pour ne jamais crasher et toujours produire un résultat :

| Situation | Comportement |
|-----------|--------------|
| Timeout LLM | 3 retries avec backoff (2s, 4s, 8s), puis retour payload stats |
| Rate limit OpenAI | Délai doublé entre retries, puis erreur propre |
| JSON invalide du LLM | Extraction regex fallback, sinon payload original |
| Erreur pendant conversation | Tente de forcer le rapport Final |
| Trop de messages (>20) | Troncature automatique de l'historique |

---

## 4. Overrides (forcer des valeurs)

Ces options permettent de forcer des valeurs au lieu de l'auto-détection.

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--override-task-type` | str | auto | Force `classification` ou `regression` |
| `--override-metric` | str | auto | Force la métrique (`f1`, `accuracy`, `rmse`, `roc_auc`) |
| `--override-feature-format` | str | auto | Force `basic`, `tags`, ou `hierarchical` |
| `--override-max-samples` | int | auto | Force le nombre d'itérations LLMFE |
| `--override-time-budget` | int | auto | Force le budget temps AutoML (secondes) |

---

## 5. Configuration AutoML

### Sélection des frameworks

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--automl-frameworks` | str | `flaml,autogluon` | Frameworks séparés par virgule |

Frameworks disponibles : `flaml`, `autogluon`, `tpot`, `h2o`

### FLAML

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--flaml-time-budget` | int | 60 | Budget temps (secondes) |
| `--flaml-metric` | str | auto | Métrique d'optimisation |

### AutoGluon

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--autogluon-presets` | str | `medium_quality_faster_train` | Preset qualité/vitesse |
| `--autogluon-time-budget` | int | 60 | Budget temps (secondes) |

Presets disponibles :
- `best_quality` : Meilleure qualité, plus lent
- `high_quality` : Haute qualité
- `good_quality` : Bon compromis
- `medium_quality_faster_train` : Rapide (défaut)
- `optimize_for_inference` : Optimisé pour inférence

### TPOT

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--tpot-generations` | int | 7 | Nombre de générations |
| `--tpot-population-size` | int | 25 | Taille de population |
| `--tpot-cv` | int | 5 | Folds cross-validation |

### H2O

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--h2o-time-budget` | int | 60 | Budget temps (secondes) |
| `--h2o-verbosity` | str | `info` | Niveau de log (debug, info, warn) |
| `--h2o-save-mojo` | flag | on | Exporter modèle MOJO |

---

## 6. Seuils d'analyse (avancé)

Ces options contrôlent les seuils de détection automatique.

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--high-cardinality-threshold` | int | 50 | Seuil haute cardinalité catégorielle |
| `--high-missing-threshold` | float | 0.3 | Seuil valeurs manquantes (ratio) |
| `--strong-corr-threshold` | float | 0.97 | Seuil corrélation pour leakage |
| `--text-unique-ratio` | float | 0.5 | Ratio unicité pour détecter texte |
| `--id-unique-ratio` | float | 0.9 | Ratio unicité pour détecter ID |

---

## 7. Performance & Parallélisation

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--num-samplers` | int | 1 | Samplers parallèles LLMFE |
| `--num-evaluators` | int | 1 | Évaluateurs parallèles LLMFE |
| `--evaluate-timeout` | int | 30 | Timeout évaluation code (sec) |
| `--n-jobs` | int | -1 | CPUs pour TPOT (-1 = tous) |

---

## Exemples

### Analyse rapide

```bash
# Analyse statistique sur Titanic
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only
```

### Réutilisation de l'analyse existante

```bash
# Première exécution : génère l'analyse
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --project titanic_v1 \
    --analyse-only

# Deuxième exécution : FE réutilise l'analyse existante (pas de regénération)
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --project titanic_v1 \
    --no-automl

# Force la regénération de l'analyse
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --project titanic_v1 \
    --no-automl \
    --force-analyse
```

### Avec analyse des corrélations

```bash
# Corrélations avec méthodes par défaut
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only \
    --with-correlations

# Corrélations + analyse LLM
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only \
    --with-correlations \
    --with-llm
```

### Avec analyse métier LLM

```bash
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --analyse-only \
    --with-llm \
    --analyse-model gpt-4o
```

### Pipeline complet avec overrides

```bash
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --full \
    --override-metric f1 \
    --override-time-budget 120 \
    --automl-frameworks flaml,autogluon
```

### Feature Engineering avancé

```bash
python tests/integration/test_pipeline_all.py \
    --dataset avis_client \
    --target label \
    --no-automl \
    --llmfe-model gpt-4 \
    --override-max-samples 15 \
    --override-feature-format hierarchical
```

### AutoML spécifique (H2O seul)

```bash
python tests/integration/test_pipeline_all.py \
    --dataset titanic \
    --target Survived \
    --full \
    --automl-frameworks h2o \
    --h2o-time-budget 300 \
    --h2o-verbosity debug
```

### Configuration des seuils d'analyse

```bash
python tests/integration/test_pipeline_all.py \
    --dataset mon_dataset \
    --target ma_cible \
    --analyse-only \
    --high-cardinality-threshold 100 \
    --high-missing-threshold 0.5
```

---

## Datasets disponibles

Les datasets doivent être placés dans `data/raw/{nom_dataset}/train.csv`.

| Dataset | Target suggéré | Description |
|---------|----------------|-------------|
| `titanic` | `Survived` | Classification binaire (survie) |
| `avis_client` | `label` | Analyse de sentiments |
| `cate_metier` | - | Catégorisation métier |
| `verbatims` | - | Textes libres |

---

## Notes

- Les paramètres `--override-*` forcent une valeur au lieu de l'auto-détection
- Si `--project` n'est pas spécifié, il prend la valeur de `--dataset`
- Les frameworks AutoML peuvent être combinés : `--automl-frameworks flaml,autogluon,h2o`
- Le budget temps est en secondes pour tous les frameworks sauf TPOT (minutes)
