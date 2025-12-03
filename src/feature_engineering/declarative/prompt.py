# prompt.py
from typing import Dict, Any, Optional, List
import textwrap

def build_prompt_from_report(
    report: Dict[str, Any],
    *,
    user_description: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    max_features_in_prompt: int = 50,
) -> str:
    """
    Construit un prompt structuré pour le LLM à partir du rapport FE.
    Adapté au format JSON fourni : "basic_stats" pour global, "target" (singulier),
    "features" comme liste.
    On injecte les infos statistiques les plus importantes :
      - type de problème (classification / régression)
      - distribution de la cible
      - rôle, NaN, cardinalité, flags pour chaque feature
    """

    # Adaptation au format JSON : global = basic_stats
    g = report["basic_stats"]

    # Target : singulier, convertir en dict {name: info}
    target_info = report["target"]
    targets = {target_info["name"]: target_info}

    # Features : liste -> dict par name
    features = {f["name"]: f for f in report["features"]}
    # Target cols pour compatibilité
    target_cols = [target_info["name"]]

    # On limite le nombre de features détaillées pour ne pas exploser le contexte
    all_feature_names = list(features.keys())
    selected_feature_names = all_feature_names[:max_features_in_prompt]

    # --- Résumé des cibles ---
    targets_summary_lines: List[str] = []
    for t_name, t_info in targets.items():
        # Adaptation : problem_type au lieu de problem_hint, inferred_target_type pour target_type
        line = (
            f"problem={t_info['problem_type']}, "
            f"n_unique={t_info['n_unique']}, "
            f"missing_rate={t_info['missing_rate']:.1%}"
        )
        targets_summary_lines.append(line)

    # --- Détail des features (sélectionnées) ---
    features_summary_lines: List[str] = []

    # Pour calculer unique_ratio = n_unique / n_rows
    n_rows = g["n_rows"]

    # Mapping simple inferred_type -> rôle générique
    role_from_inferred = {
        "numeric": "numeric",
        "categorical_low": "categorical",
        "categorical_high": "categorical",
        "text": "text",
        "datetime": "datetime",
        "boolean": "boolean",
    }

    for fname in selected_feature_names:
        finfo = features[fname]

        # Flags : liste dans JSON, mapper aux indicateurs
        flags_list = finfo.get("flags", [])
        flags = []
        if "CONSTANT" in flags_list or "CONST" in flags_list:
            flags.append("CONST")
        if "ID_LIKE" in flags_list:
            flags.append("ID_LIKE")
        if "HIGH_CARDINALITY" in flags_list:
            flags.append("HIGH_CARD")

        flags_str = f" flags={','.join(flags)}" if flags else ""

        # dtype : on prend pandas_dtype si dispo, sinon inferred_type, sinon "unknown"
        dtype = finfo.get("pandas_dtype", finfo.get("inferred_type", "unknown"))

        # rôle : si pas présent dans le JSON, on le déduit de inferred_type
        inferred_type = finfo.get("inferred_type", "unknown")
        role = finfo.get("role", role_from_inferred.get(inferred_type, "unknown"))

        # n_unique, unique_ratio, missing_rate : adaptés au JSON
        n_unique = finfo.get("n_unique", 0)
        unique_ratio = n_unique / n_rows if n_rows else 0.0
        missing_rate = finfo.get("missing_rate", 0.0)

        line = (
            f"- {fname}: role={role}, dtype={dtype}, "
            f"n_unique={n_unique}, "
            f"unique_ratio={unique_ratio:.1%}, "
            f"missing_rate={missing_rate:.1%}{flags_str}"
        )
        features_summary_lines.append(line)

    # --- Instructions sur le format de sortie JSON ---
    json_spec = textwrap.dedent(
        """
        Tu dois répondre UNIQUEMENT avec un JSON valide, sans texte autour.
        Le JSON doit avoir la structure suivante :

        {
          "features_plan": [
            {
              "name": "nom_de_la_feature_creee_ou_modifiee",
              "type": "numeric_derived | categorical_encoding | text_embedding | datetime_derived | other",
              "inputs": ["col1", "col2"],
              "transformation": "formule_présent_dans_FORMULE",
              "descriptions_transformations": "description_symbolique_de_la_transformation_ou_formule",
              "encoding": "one_hot | target_encoding | ordinal | hashing | None",
              "reason": "raison_métier_ou_statistique_qui_explique_ce_choix"
            }
          ],
          "global_notes": [
            "recommandation_globale_1",
            "recommandation_globale_2"
          ],
          "questions_for_user": [
            "question_1",
            "question_2"
          ]
        }

        - "features_plan" : liste de transformations de features concrètes à créer
          ou d'encodages à appliquer.
        - "global_notes" : remarques générales sur le feature engineering à appliquer.
        - "questions_for_user" : questions à poser à l'utilisateur pour affiner encore
          le choix des features (ex: sémantique métier, contraintes, etc.).
        """
    )

    # --- Instructions sur le rôle du LLM ---
    system_instructions = textwrap.dedent(
        """
        Tu es un assistant expert en machine learning et en feature engineering.
        On te fournit un résumé statistique d'un dataset (cibles + features).
        Ton objectif est de proposer un plan de feature engineering intelligent
        qui exploite ces informations statistiques.

        Tu dois :
        - Utiliser le type de problème (classification / régression) pour adapter
          tes suggestions (encodage de la cible, gestion de l'imbalance, etc.).
        - Utiliser le rôle des features (numeric / categorical / text / datetime / boolean)
          pour proposer des transformations adaptées.
        - Tenir compte des taux de valeurs manquantes, de la cardinalité, et des flags
          (ID_LIKE, HIGH_CARD, CONST) pour éviter les fuites de données et le sur-apprentissage.
        - Proposer des features dérivées métierment pertinentes lorsque c'est possible
          (ratios, agrégations temporelles, etc.).
        - Proposer des encodages adaptés aux catégorielles (one-hot vs target encoding, etc.).
        - Proposer des représentations pour le texte (TF-IDF, embeddings, etc.) si pertinent.

        Tu dois utiliser uniquement les transformations suivantes :

        FORMULE:

        NUMERIC (input = x)
        - scaling : standard(x), robust(x)
        - imputation : impute_mean(x), impute_median(x)
        - unary nonlinear : log1p(x), sqrt(x), square(x), cube(x)
        - binning : bin_quantile(x, N), bin_uniform(x, N)
        - clipping : clip(x, q01, q99)
        - normalization : zscore(x)

        NUMERIC × NUMERIC (inputs = x, y)
        - sum : x + y
        - diff : x - y
        - prod : x * y
        - ratio : x / (y + 1e-6)
        - min_max : min(x,y), max(x,y)

        CATEGORICAL (input = col)
        - one_hot(col)
        - target_encoding(col)
        - ordinal(col)
        - hashing(col, n_components=32)
        - reduce_cardinality(col, strategy="frequency" | "first_letter")
        - freq_encoding(col)

        TEXT (input = text)
        - tfidf(text)
        - hashing_ngrams(text)
        - embedding(text, model="mistral" | "e5-base")

        DATETIME (input = date)
        - year(date), month(date), day(date), dayofweek(date)
        - hour(date), minute(date)
        - cyclic_encoding(date.month)
        - time_since(date)

        """
    )

    if extra_instructions:
        system_instructions += "\nInstructions supplémentaires :\n" + extra_instructions.strip()

    # --- Description utilisateur (optionnelle) ---
    user_desc_block = ""
    if user_description:
        user_desc_block = "Description donnée par l'utilisateur :\n" + user_description.strip() + "\n"

    # --- Assemblage final du prompt ---
    prompt = textwrap.dedent(
        f"""
        {system_instructions}

        ========================
        CONTEXTE DU DATASET
        ========================

        Nombre de lignes : {g['n_rows']}
        Nombre de features : {g['n_features']}
        Cibles : {target_cols}

        {user_desc_block}

        ------------------------
        ANALYSE DES CIBLES
        ------------------------
        {chr(10).join(targets_summary_lines)}

        ------------------------
        ANALYSE DES FEATURES (sélection)
        ------------------------
        (Maximum de {max_features_in_prompt} features listées)

        {chr(10).join(features_summary_lines)}

        ========================
        FORMAT DE RÉPONSE ATTENDU
        ========================
        {json_spec}

        Rappels importants :
        - Ne propose pas de features qui utilisent directement la cible.
        - Ne propose pas d'utiliser des variables ID_LIKE comme features brutes.
        - Si une feature est de haute cardinalité, préfère des encodages compacts
          (target encoding, hashing, embeddings) plutôt que du one-hot naïf.
        - Tu peux utiliser les informations de rôle, de NaN, de cardinalité, et
          de flags pour justifier tes choix dans "reason".
        - Répond uniquement en francais (nom de features, explication)

        Réponds uniquement avec le JSON.
        """
    ).strip()

    return prompt
