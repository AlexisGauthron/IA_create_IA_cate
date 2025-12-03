"""
Interface Streamlit pour le Pipeline ML complet.
Intègre: Analyse -> Feature Engineering (LLMFE) -> AutoML

Usage:
    streamlit run src/front/pipeline_streamlit.py
"""

import sys
from pathlib import Path

# Ajouter le dossier racine au path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# =============================================================================
# IMPORTANT: Charger le .env AVANT tout import de src.core.config
# =============================================================================
from dotenv import load_dotenv

env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)  # override=True force le rechargement
    print(f"[STREAMLIT] .env chargé depuis {env_path}")
    print(f"[STREAMLIT] OPENAI_API_KEY présente: {os.getenv('OPENAI_API_KEY') is not None}")

    # LLMFE utilise API_KEY comme nom de variable (pas OPENAI_API_KEY)
    # On copie la valeur pour que LLMFE puisse fonctionner
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        os.environ['API_KEY'] = openai_key
        print(f"[STREAMLIT] API_KEY configurée pour LLMFE")
else:
    print(f"[STREAMLIT] ATTENTION: .env non trouvé à {env_path}")

# Maintenant on peut importer settings (qui utilisera les variables d'env déjà chargées)
from src.core.config import settings  # noqa: E402

# Debug: Vérifier que la clé est bien chargée
print(f"[DEBUG] OpenAI configuré: {settings.is_configured('openai')}")

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, List
import json
import time

from src.front.css import CUSTOM_CSS
from src.front.ui_helper import card_block


# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title="ML Pipeline - IA Create IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================
def init_session_state():
    """Initialise les variables de session."""
    defaults = {
        # Données
        "df_train": None,
        "df_test": None,
        "target_col": None,
        "project_name": "",

        # Étape courante
        "current_step": 0,  # 0=Upload, 1=Config, 2=Analyse, 3=Chat, 4=FE, 5=AutoML, 6=Results

        # Agent métier
        "chat_history": [],
        "agent_context": {},
        "analysis_complete": False,

        # Résultats
        "stats_report": None,
        "correlations_df": None,  # DataFrame des corrélations features/target
        "fe_results": None,
        "automl_results": None,

        # Configuration LLM Agent Métier
        "agent_provider": "openai",
        "agent_model": "gpt-3.5-turbo",

        # Configuration LLM Feature Engineering
        "fe_provider": "openai",
        "fe_model": "gpt-4o-mini",

        # Autres configs
        "is_regression": False,
        "max_fe_samples": 10,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# Composants UI
# =============================================================================

def render_header():
    """Affiche le header principal."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
            🤖 ML Pipeline Assistant
        </h1>
        <p style="color: var(--text-muted); font-size: 1.1rem;">
            Analyse → Feature Engineering → AutoML
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_progress_bar():
    """Affiche la barre de progression du pipeline."""
    steps = [
        ("📁", "Upload"),
        ("⚙️", "Config"),
        ("📊", "Analyse"),
        ("💬", "Agent"),
        ("🔧", "Features"),
        ("🤖", "AutoML"),
        ("📈", "Résultats"),
    ]

    current = st.session_state.current_step

    cols = st.columns(len(steps))
    for i, (icon, label) in enumerate(steps):
        with cols[i]:
            if i < current:
                color = "#22c55e"  # Vert - complété
                opacity = 1
            elif i == current:
                color = "#3b82f6"  # Bleu - actuel
                opacity = 1
            else:
                color = "#6b7280"  # Gris - à venir
                opacity = 0.5

            st.markdown(f"""
            <div style="text-align: center; opacity: {opacity};">
                <div style="font-size: 1.5rem; color: {color};">{icon}</div>
                <div style="font-size: 0.75rem; color: {color};">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)


def render_sidebar():
    """Affiche la sidebar avec les informations du projet."""
    with st.sidebar:
        st.markdown("### 📋 Projet")

        if st.session_state.project_name:
            st.markdown(f"**Nom:** {st.session_state.project_name}")

        if st.session_state.df_train is not None:
            df = st.session_state.df_train
            st.markdown(f"**Lignes:** {len(df):,}")
            st.markdown(f"**Colonnes:** {len(df.columns)}")

            if st.session_state.target_col:
                st.markdown(f"**Cible:** `{st.session_state.target_col}`")
                task_type = "Régression" if st.session_state.is_regression else "Classification"
                st.markdown(f"**Type:** {task_type}")

        st.markdown("---")

        # === Configuration Agent Métier ===
        st.markdown("### 💬 Agent Métier")
        st.session_state.agent_provider = st.selectbox(
            "Provider (Agent)",
            ["openai", "ollama"],
            index=0 if st.session_state.agent_provider == "openai" else 1,
            key="agent_provider_select",
        )

        if st.session_state.agent_provider == "openai":
            agent_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        else:
            agent_models = ["llama3.1", "mistral", "codellama"]

        current_agent_idx = agent_models.index(st.session_state.agent_model) if st.session_state.agent_model in agent_models else 0
        st.session_state.agent_model = st.selectbox(
            "Modèle (Agent)",
            agent_models,
            index=current_agent_idx,
            key="agent_model_select",
        )

        st.markdown("---")

        # === Configuration Feature Engineering ===
        st.markdown("### 🔧 Feature Engineering")
        st.session_state.fe_provider = st.selectbox(
            "Provider (FE)",
            ["openai", "ollama"],
            index=0 if st.session_state.fe_provider == "openai" else 1,
            key="fe_provider_select",
        )

        if st.session_state.fe_provider == "openai":
            fe_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        else:
            fe_models = ["llama3.1", "mistral", "codellama"]

        current_fe_idx = fe_models.index(st.session_state.fe_model) if st.session_state.fe_model in fe_models else 0
        st.session_state.fe_model = st.selectbox(
            "Modèle (FE)",
            fe_models,
            index=current_fe_idx,
            key="fe_model_select",
        )

        st.markdown("---")

        # Bouton reset
        if st.button("🔄 Nouveau projet", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# =============================================================================
# Étape 0: Upload des données
# =============================================================================

def render_upload_step():
    """Étape d'upload des fichiers CSV."""
    st.markdown("## 📁 Étape 1: Charger les données")

    with card_block():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Fichier d'entraînement (requis)")
            train_file = st.file_uploader(
                "Choisir train.csv",
                type=["csv"],
                key="train_upload",
            )

            if train_file is not None:
                # Détecter le séparateur
                sep = detect_separator(train_file)
                train_file.seek(0)

                df_train = pd.read_csv(train_file, sep=sep)
                st.session_state.df_train = df_train

                st.success(f"✅ {len(df_train)} lignes, {len(df_train.columns)} colonnes")

                with st.expander("Aperçu des données"):
                    st.dataframe(df_train.head(10), use_container_width=True)

        with col2:
            st.markdown("### Fichier de test (optionnel)")
            test_file = st.file_uploader(
                "Choisir test.csv",
                type=["csv"],
                key="test_upload",
            )

            if test_file is not None:
                sep = detect_separator(test_file)
                test_file.seek(0)

                df_test = pd.read_csv(test_file, sep=sep)
                st.session_state.df_test = df_test

                st.success(f"✅ {len(df_test)} lignes, {len(df_test.columns)} colonnes")

    # Bouton suivant
    if st.session_state.df_train is not None:
        if st.button("Suivant →", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()


def detect_separator(file) -> str:
    """Détecte le séparateur d'un fichier CSV."""
    first_line = file.readline().decode('utf-8')
    file.seek(0)

    candidates = {',': 0, ';': 0, '\t': 0, '|': 0}
    for sep in candidates:
        candidates[sep] = first_line.count(sep)

    best_sep = max(candidates, key=candidates.get)
    return best_sep if candidates[best_sep] > 0 else ','


# =============================================================================
# Étape 1: Configuration
# =============================================================================

def render_config_step():
    """Étape de configuration du projet."""
    st.markdown("## ⚙️ Étape 2: Configuration")

    df = st.session_state.df_train

    with card_block():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Informations du projet")

            project_name = st.text_input(
                "Nom du projet",
                value=st.session_state.project_name or "mon_projet",
                help="Identifiant unique pour ce projet",
            )
            st.session_state.project_name = project_name

            target_col = st.selectbox(
                "Colonne cible",
                options=list(df.columns),
                index=len(df.columns) - 1,  # Dernière colonne par défaut
                help="La variable à prédire",
            )
            st.session_state.target_col = target_col

        with col2:
            st.markdown("### Type de problème")

            # Détection automatique
            target_values = df[target_col].nunique()
            auto_regression = target_values > 20 and df[target_col].dtype in ['int64', 'float64']

            is_regression = st.radio(
                "Type de tâche",
                options=[False, True],
                format_func=lambda x: "Régression" if x else "Classification",
                index=1 if auto_regression else 0,
                horizontal=True,
            )
            st.session_state.is_regression = is_regression

            if not is_regression:
                st.info(f"📊 {target_values} classes détectées dans `{target_col}`")
            else:
                st.info(f"📈 Variable continue: {df[target_col].min():.2f} - {df[target_col].max():.2f}")

            st.markdown("### Feature Engineering")
            max_samples = st.slider(
                "Nombre max d'itérations LLMFE",
                min_value=5,
                max_value=100,
                value=st.session_state.max_fe_samples,
                step=5,
                help="Plus d'itérations = meilleurs résultats mais plus long",
            )
            st.session_state.max_fe_samples = max_samples

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Retour", use_container_width=True):
            st.session_state.current_step = 0
            st.rerun()
    with col2:
        if st.button("Suivant →", type="primary", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()


# =============================================================================
# Étape 2: Analyse statistique
# =============================================================================

def render_analyse_step():
    """Étape d'analyse statistique automatique."""
    st.markdown("## 📊 Étape 3: Analyse du dataset")

    if st.session_state.stats_report is None:
        with card_block():
            st.markdown("### Lancer l'analyse statistique")
            st.markdown("""
            Cette étape génère un rapport complet sur votre dataset:
            - Types de colonnes et valeurs manquantes
            - Statistiques descriptives
            - Corrélations avec la cible
            - Recommandations de feature engineering
            """)

            if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
                run_statistical_analysis()
    else:
        render_analysis_results()

    # Navigation
    if st.session_state.stats_report is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Retour", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("Continuer vers l'agent →", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()


def run_statistical_analysis():
    """Exécute l'analyse statistique et les corrélations."""
    import src.analyse.statistiques.report as report
    from src.analyse.path_config import AnalysePathConfig

    progress = st.progress(0, text="Initialisation...")

    try:
        df = st.session_state.df_train
        target = st.session_state.target_col

        # Créer le path_config (crée automatiquement les dossiers)
        path_config = AnalysePathConfig(project_name=st.session_state.project_name)
        print(f"[ANALYSE] Dossier créé: {path_config.project_dir}")

        progress.progress(20, text="Analyse des colonnes...")
        time.sleep(0.5)

        progress.progress(50, text="Calcul des statistiques...")

        report_data = report.analyze_dataset_for_fe(
            df,
            target_cols=target,
            print_report=False,
            dataset_name=st.session_state.project_name,
            business_description=f"Analyse du projet {st.session_state.project_name}",
        )

        progress.progress(70, text="Génération du rapport...")

        st.session_state.stats_report = report_data.get("llm_payload", report_data)
        st.session_state.analysis_complete = True

        # Sauvegarder le rapport sur disque
        path_config.save_stats_report(st.session_state.stats_report)
        print(f"[ANALYSE] Rapport sauvegardé: {path_config.stats_report_path}")

        # Calcul des corrélations features/target
        progress.progress(85, text="Calcul des corrélations...")
        try:
            from src.analyse.correlation.correlation import FeatureCorrelationAnalyzer
            import traceback as tb

            # Déterminer le type de tâche
            task = "regression" if st.session_state.is_regression else "classification"
            print(f"[CORRELATIONS] Démarrage calcul - target={target}, task={task}")

            # Créer l'analyseur et calculer le score combiné
            analyzer = FeatureCorrelationAnalyzer(df, target_col=target, task=task)
            correlations_df = analyzer.combined_feature_score(normalize=True)

            st.session_state.correlations_df = correlations_df
            print(f"[CORRELATIONS] ✅ Corrélations calculées: {len(correlations_df)} features")

        except Exception as corr_error:
            # Ne pas bloquer si les corrélations échouent
            import traceback as tb
            error_details = tb.format_exc()
            print(f"[CORRELATIONS] ⚠️ Erreur: {corr_error}")
            print(f"[CORRELATIONS] Traceback:\n{error_details}")
            st.session_state.correlations_df = None
            # Afficher un warning dans l'interface
            st.warning(f"⚠️ Corrélations non calculées: {corr_error}")

        progress.progress(100, text="Terminé!")
        time.sleep(0.3)
        progress.empty()

        st.success("✅ Analyse terminée!")
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_analysis_results():
    """Affiche les résultats de l'analyse."""
    report = st.session_state.stats_report

    # Extraire les données de la structure llm_payload
    # Structure: {context, basic_stats, target, features (liste), analysis_config, global_notes}
    basic_stats = report.get("basic_stats", {})
    features_list = report.get("features", [])
    target_info = report.get("target", {})

    with card_block():
        st.markdown("### ✅ Analyse complétée")

        # Résumé depuis basic_stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_features = basic_stats.get("n_features", len(features_list))
            st.metric("Features", n_features)

        with col2:
            # Compter les features avec valeurs manquantes
            missing = sum(
                1 for feat in features_list
                if feat.get("missing_ratio", 0) > 0
            )
            st.metric("Avec manquants", missing)

        with col3:
            n_cat = basic_stats.get("n_categorical_features", 0)
            st.metric("Catégorielles", n_cat)

        with col4:
            n_num = basic_stats.get("n_numeric_features", 0)
            st.metric("Numériques", n_num)

    # Infos sur la cible
    if target_info:
        with card_block():
            st.markdown("### 🎯 Variable cible")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Nom:** `{target_info.get('name', 'N/A')}`")

            with col2:
                task_type = target_info.get("task_type", "classification")
                st.markdown(f"**Type:** {task_type}")

            with col3:
                n_classes = target_info.get("n_classes")
                if n_classes:
                    st.markdown(f"**Classes:** {n_classes}")

    # Détails par feature
    with st.expander("📋 Détails par feature"):
        if features_list:
            columns_data = []
            for feat in features_list:
                columns_data.append({
                    "Feature": feat.get("name", "?"),
                    "Type": feat.get("inferred_type", feat.get("dtype", "?")),
                    "Uniques": feat.get("n_unique", 0),
                    "Manquants %": f"{feat.get('missing_ratio', 0)*100:.1f}%",
                    "Flags": ", ".join(feat.get("flags", [])) or "-",
                })

            st.dataframe(pd.DataFrame(columns_data), use_container_width=True)
        else:
            st.info("Aucune feature analysée.")

    # Section Corrélations
    correlations_df = st.session_state.get("correlations_df")
    if correlations_df is not None and len(correlations_df) > 0:
        with card_block():
            st.markdown("### 📈 Corrélations avec la cible")
            st.markdown("""
            Analyse multi-métriques de la relation entre chaque feature et la variable cible.
            Plus le score combiné est élevé, plus la feature est potentiellement prédictive.
            """)

            # Graphique des top 15 features par score combiné
            top_n = min(15, len(correlations_df))
            top_features = correlations_df.head(top_n).copy()

            # Créer le graphique en barres
            chart_data = top_features[["feature", "combined_score"]].set_index("feature")
            chart_data = chart_data.sort_values("combined_score", ascending=True)  # Pour affichage horizontal

            st.markdown(f"#### Top {top_n} features par score combiné")
            st.bar_chart(chart_data, horizontal=True, use_container_width=True)

            # Tableau détaillé avec toutes les métriques
            with st.expander("🔍 Tableau détaillé des corrélations"):
                # Formater les colonnes pour un affichage plus lisible
                display_df = correlations_df.copy()

                # Renommer les colonnes disponibles
                rename_map = {
                    "feature": "Feature",
                    "pearson": "Pearson",
                    "spearman": "Spearman",
                    "kendall": "Kendall",
                    "mutual_info": "Mutual Info",
                    "combined_score": "Score Combiné",
                }
                # Ajouter MIC et PhiK seulement si présents
                if "mic" in display_df.columns:
                    rename_map["mic"] = "MIC"
                if "phik" in display_df.columns:
                    rename_map["phik"] = "PhiK"

                display_df = display_df.rename(columns=rename_map)

                # Formater seulement les colonnes présentes
                format_dict = {"Score Combiné": "{:.3f}"}
                for col in ["Pearson", "Spearman", "Kendall", "Mutual Info", "MIC", "PhiK"]:
                    if col in display_df.columns:
                        format_dict[col] = "{:.3f}"

                # Afficher avec formatage
                st.dataframe(
                    display_df.style.format(format_dict).background_gradient(subset=["Score Combiné"], cmap="YlGn"),
                    use_container_width=True,
                    hide_index=True,
                )

                # Légende des métriques (adaptée aux colonnes disponibles)
                legend = """
                **Légende des métriques:**
                - **Pearson**: Corrélation linéaire (sensible aux outliers)
                - **Spearman**: Corrélation de rang (monotonie)
                - **Kendall**: Corrélation de rang (robuste)
                - **Mutual Info**: Information mutuelle (dépendances non-linéaires)
                """
                if "MIC" in display_df.columns:
                    legend += "- **MIC**: Maximal Information Coefficient (relations complexes)\n"
                if "PhiK" in display_df.columns:
                    legend += "- **PhiK**: Corrélation catégoriel/numérique robuste\n"
                st.markdown(legend)

    # Notes globales
    global_notes = report.get("global_notes", [])
    if global_notes:
        with st.expander("📝 Notes"):
            for note in global_notes:
                st.markdown(f"- {note}")

    # JSON complet
    with st.expander("🔍 Rapport JSON complet"):
        st.json(report)


# =============================================================================
# Étape 3: Agent métier (Chat)
# =============================================================================

def parse_llm_response(response: str) -> dict:
    """
    Parse la réponse JSON du LLM et extrait la question.

    Le LLM répond en JSON: {"Mode": "Question", "Q": "..."}
    """
    try:
        data = json.loads(response)
        return {
            "mode": data.get("Mode", data.get("mode", "Question")),
            "question": data.get("Q", data.get("q", data.get("question", response))),
            "raw": response,
        }
    except json.JSONDecodeError:
        # Si ce n'est pas du JSON, retourner tel quel
        return {
            "mode": "Question",
            "question": response,
            "raw": response,
        }


def format_assistant_message(content: str) -> str:
    """Formate le message assistant pour l'affichage."""
    parsed = parse_llm_response(content)
    return parsed["question"]


def render_chat_step():
    """Étape de conversation avec l'agent métier."""
    st.markdown("## 💬 Étape 4: Clarification métier")

    st.markdown("""
    L'agent va vous poser des questions pour mieux comprendre votre problème métier.
    Vos réponses enrichiront le feature engineering.

    **Commandes disponibles:** `skip` (passer), `done` (terminer)
    """)

    # Zone de chat
    chat_container = st.container()

    with chat_container:
        # Afficher l'historique
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    # Parser et afficher uniquement la question
                    formatted = format_assistant_message(content)
                    st.markdown(formatted)
            else:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)

    # Démarrer la conversation si vide
    if not st.session_state.chat_history:
        if st.button("🚀 Démarrer la conversation", type="primary"):
            start_agent_conversation()

    # Input utilisateur
    if st.session_state.chat_history:
        user_input = st.chat_input("Votre réponse...")

        if user_input:
            process_user_input(user_input)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Retour", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col2:
        if st.button("Passer au Feature Engineering →", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()


def start_agent_conversation():
    """Démarre la conversation avec l'agent."""
    # Afficher l'animation de chargement
    with st.spinner("🤖 L'agent analyse votre dataset..."):
        try:
            from src.analyse.metier.chatbot_llm import BusinessClarificationBot
            from src.core.llm_client import OllamaClient

            # Créer le client LLM (utilise la config Agent)
            provider = st.session_state.agent_provider
            model = st.session_state.agent_model

            llm_client = OllamaClient(
                model=model,
                provider=provider,
            )

            # Créer le chatbot avec le rapport statistique
            stats = st.session_state.stats_report
            chatbot = BusinessClarificationBot(
                stats=stats,
                llm=llm_client,
            )

            # Premier appel pour obtenir la première question
            response = chatbot.ask_next(user_answer=None)

            # Stocker le chatbot dans la session
            st.session_state.chatbot = chatbot

            # Ajouter à l'historique
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
            })

        except Exception as e:
            st.error(f"❌ Erreur au démarrage de l'agent: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Recharger la page pour afficher la conversation
    st.rerun()


def process_user_input(user_input: str):
    """Traite l'input utilisateur dans le chat."""
    # Ajouter le message utilisateur
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
    })

    # Commandes spéciales
    if user_input.lower().strip() in ["skip", "passer"]:
        with st.spinner("🤖 Question suivante..."):
            chatbot = st.session_state.get("chatbot")
            if chatbot:
                try:
                    response = chatbot.ask_next("L'utilisateur a choisi de passer cette question.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                    })
                except Exception:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Question passée.",
                    })
        st.rerun()
        return

    if user_input.lower().strip() in ["done", "terminé", "fini"]:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Parfait! Je vais utiliser les informations collectées pour le feature engineering.",
        })
        st.session_state.current_step = 4
        st.rerun()
        return

    # Envoyer au chatbot avec animation de chargement
    chatbot = st.session_state.get("chatbot")
    if chatbot is None:
        st.error("Session chatbot perdue. Veuillez redémarrer la conversation.")
        return

    with st.spinner("🤖 L'agent réfléchit..."):
        try:
            response = chatbot.ask_next(user_answer=user_input)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
            })

            # Vérifier si le LLM est en mode Final
            parsed = parse_llm_response(response)
            if parsed["mode"].lower() == "final":
                st.session_state.current_step = 4

        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            return

    st.rerun()


# =============================================================================
# Étape 4: Feature Engineering
# =============================================================================

def render_fe_step():
    """Étape de feature engineering automatique."""
    st.markdown("## 🔧 Étape 5: Feature Engineering (LLMFE)")

    with card_block():
        st.markdown("""
        LLMFE va générer automatiquement des features optimisées pour votre problème.

        **Configuration:**
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **Provider:** {st.session_state.fe_provider}")
            st.markdown(f"- **Modèle:** {st.session_state.fe_model}")
            st.markdown(f"- **Itérations max:** {st.session_state.max_fe_samples}")
        with col2:
            task = "Régression" if st.session_state.is_regression else "Classification"
            st.markdown(f"- **Type:** {task}")
            st.markdown(f"- **Cible:** `{st.session_state.target_col}`")

    if st.session_state.fe_results is None:
        if st.button("🚀 Lancer le Feature Engineering", type="primary", use_container_width=True):
            run_feature_engineering()
    else:
        render_fe_results()

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Retour", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col2:
        # Option pour sauter le FE et aller directement à AutoML
        if st.session_state.fe_results is None:
            if st.button("Passer → AutoML", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()
    with col3:
        if st.session_state.fe_results is not None:
            if st.button("Continuer vers AutoML →", type="primary", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()


def run_feature_engineering():
    """Lance le feature engineering LLMFE avec affichage progressif en temps réel."""
    from src.front.fe_runner_async import AsyncFERunner
    from src.front.fe_progress_monitor import (
        get_current_progress,
        get_metrics,
        get_chart_data,
        get_recent_samples,
        load_final_summary,
        load_best_model_json,
    )
    from pathlib import Path

    st.markdown("### 🔧 Feature Engineering en cours...")
    st.info(f"📊 Configuration: {st.session_state.fe_provider} / {st.session_state.fe_model} | Max: {st.session_state.max_fe_samples} itérations")

    # Variables pour le suivi
    max_samples = st.session_state.max_fe_samples

    # Créer et lancer le runner asynchrone
    async_runner = AsyncFERunner(project_name=st.session_state.project_name)

    # Placeholders pour l'affichage dynamique (créés AVANT le lancement)
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    recent_placeholder = st.empty()

    # État initial
    with status_placeholder.container():
        st.warning("⚠️ Démarrage de LLMFE... Cette opération peut prendre plusieurs minutes.")

    with progress_placeholder.container():
        st.progress(0.0, text="🚀 Initialisation...")

    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Meilleur", "—")
        with col2:
            st.metric("✅ Valides", 0)
        with col3:
            st.metric("❌ Échouées", 0)
        with col4:
            st.metric("📊 Succès", "0%")

    try:
        # Lancer LLMFE en arrière-plan
        async_runner.start(
            df_train=st.session_state.df_train,
            target_col=st.session_state.target_col,
            is_regression=st.session_state.is_regression,
            max_samples=max_samples,
            api_model=st.session_state.fe_model,
            use_api=(st.session_state.fe_provider == "openai"),
        )

        samples_dir = async_runner.get_samples_dir()
        results_dir = async_runner.get_results_dir()

        with status_placeholder.container():
            st.info(f"🔄 LLMFE en cours... Les résultats apparaîtront ci-dessous.")

        # Boucle de polling - mise à jour toutes les 2 secondes
        last_count = 0
        while async_runner.is_running():
            time.sleep(2)  # Polling toutes les 2 secondes

            # Lire les résultats actuels depuis les fichiers JSON
            df = get_current_progress(samples_dir)
            current_count = len(df)

            # Mettre à jour seulement s'il y a de nouveaux résultats
            if current_count > 0:
                metrics = get_metrics(df)

                # Mettre à jour la barre de progression
                with progress_placeholder.container():
                    progress = min(current_count / max_samples, 1.0)
                    st.progress(progress, text=f"⏳ Itération {current_count}/{max_samples}")

                # Mettre à jour les métriques
                with metrics_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        best = metrics["best_score"]
                        best_str = f"{best:.4f}" if best is not None else "—"
                        st.metric("🏆 Meilleur", best_str)
                    with col2:
                        st.metric("✅ Valides", metrics["valid"])
                    with col3:
                        st.metric("❌ Échouées", metrics["failed"])
                    with col4:
                        st.metric("📊 Succès", f"{metrics['success_rate']:.0f}%")

                # Mettre à jour le graphique
                with chart_placeholder.container():
                    chart_df = get_chart_data(df)
                    if len(chart_df) > 1:
                        st.markdown("#### 📈 Évolution du score")
                        st.line_chart(chart_df, use_container_width=True)

                # Afficher les samples récents
                with recent_placeholder.container():
                    recent = get_recent_samples(df, n=5)
                    if recent:
                        st.markdown("#### 🕐 Derniers samples")
                        recent_text = " | ".join([
                            f"{s['status']} #{s['sample_order']}: {s['score']:.4f}" if s['score'] else f"{s['status']} #{s['sample_order']}: échec"
                            for s in recent
                        ])
                        st.caption(recent_text)

                last_count = current_count

        # Vérifier s'il y a eu une erreur
        if async_runner.has_error():
            status_placeholder.empty()
            st.error(f"❌ Erreur LLMFE: {async_runner.get_error()}")
            import traceback
            st.code(traceback.format_exc())
            return

        # LLMFE terminé avec succès - affichage final
        with progress_placeholder.container():
            st.progress(1.0, text="✅ Feature Engineering terminé!")

        # Charger les résultats finaux
        final_summary = load_final_summary(results_dir)
        if final_summary:
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    best = final_summary.get("best_score", 0)
                    st.metric("🏆 Meilleur", f"{best:.4f}")
                with col2:
                    st.metric("✅ Valides", final_summary.get("valid_samples", 0))
                with col3:
                    st.metric("❌ Échouées", final_summary.get("failed_samples", 0))
                with col4:
                    total = final_summary.get("total_samples", 1)
                    valid = final_summary.get("valid_samples", 0)
                    rate = (valid / total * 100) if total > 0 else 0
                    st.metric("📊 Succès", f"{rate:.0f}%")

        # Afficher le graphique final
        df = get_current_progress(samples_dir)
        with chart_placeholder.container():
            chart_df = get_chart_data(df)
            if len(chart_df) > 0:
                st.markdown("#### 📈 Évolution finale du score")
                st.line_chart(chart_df, use_container_width=True)

        status_placeholder.empty()
        recent_placeholder.empty()

        # Sauvegarder les résultats dans la session
        result = async_runner.get_result()
        if result:
            st.session_state.fe_results = result
        else:
            # Fallback si result est None
            st.session_state.fe_results = {
                "results_dir": str(results_dir),
                "samples_dir": str(samples_dir),
            }

        st.success("✅ Feature Engineering terminé avec succès!")
        time.sleep(1)
        st.rerun()

    except Exception as e:
        progress_placeholder.empty()
        metrics_placeholder.empty()
        chart_placeholder.empty()
        status_placeholder.empty()
        recent_placeholder.empty()
        st.error(f"❌ Erreur LLMFE: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_fe_results():
    """Affiche le dashboard complet des résultats LLMFE."""
    results = st.session_state.fe_results
    results_dir = Path(results.get('results_dir', ''))

    # Charger les données
    summary = _load_json(results_dir / "summary.json")
    best_model = _load_json(results_dir / "best_model.json")
    all_scores = _load_json(results_dir / "all_scores.json")

    # Charger les samples
    samples = []
    samples_dir = results_dir.parent / "samples"
    if samples_dir.exists():
        for sample_file in sorted(samples_dir.glob("*.json")):
            try:
                with open(sample_file) as f:
                    samples.append(json.load(f))
            except Exception:
                pass

    # === Métriques principales ===
    st.markdown("### 📊 Résultats Feature Engineering")

    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            best_score = summary.get("best_score", 0)
            st.metric("🏆 Meilleur Score", f"{best_score:.4f}")
        with col2:
            st.metric("🔄 Total Samples", summary.get("total_samples", 0))
        with col3:
            valid = summary.get("valid_samples", 0)
            total = summary.get("total_samples", 1)
            rate = (valid / total * 100) if total > 0 else 0
            st.metric("✅ Taux Succès", f"{rate:.0f}%")
        with col4:
            total_time = summary.get("total_sample_time", 0) + summary.get("total_evaluate_time", 0)
            st.metric("⏱️ Temps Total", f"{total_time:.0f}s")

    # === Onglets ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Meilleur modèle",
        "📈 Évolution",
        "📋 Tous les essais",
        "📊 Statistiques"
    ])

    # --- Onglet 1: Meilleur modèle ---
    with tab1:
        if best_model:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Informations")
                st.metric("Score", f"{best_model.get('score', 0):.4f}")
                st.metric("Sample #", best_model.get("sample_order", "?"))

            with col2:
                st.markdown("#### Code de transformation")
                code = best_model.get("function_code", "# Pas de code disponible")
                st.code(code, language="python")

                # Bouton télécharger
                st.download_button(
                    "📋 Télécharger le code",
                    data=code,
                    file_name="best_features.py",
                    mime="text/plain",
                )
        else:
            st.info("Aucun modèle valide généré.")

    # --- Onglet 2: Évolution ---
    with tab2:
        if all_scores:
            df = pd.DataFrame(all_scores)

            if "sample_order" in df.columns and "score" in df.columns:
                df = df.sort_values("sample_order")

                # Calculer le meilleur score cumulatif
                best_so_far = []
                current_best = float('-inf')
                for score in df["score"]:
                    current_best = max(current_best, score)
                    best_so_far.append(current_best)
                df["Meilleur cumulé"] = best_so_far
                df = df.rename(columns={"score": "Score"})

                st.markdown("#### Évolution du score")
                st.line_chart(
                    df.set_index("sample_order")[["Score", "Meilleur cumulé"]],
                    use_container_width=True,
                )

                if best_model:
                    best_sample = best_model.get("sample_order")
                    if best_sample:
                        st.success(f"🎯 Meilleur score atteint à l'itération #{best_sample}")
        else:
            st.info("Données d'évolution non disponibles.")

    # --- Onglet 3: Tous les essais ---
    with tab3:
        if samples:
            # Créer le DataFrame
            data = []
            for sample in samples:
                score = sample.get("score")
                data.append({
                    "#": sample.get("sample_order", "?"),
                    "Score": f"{score:.4f}" if score is not None else "—",
                    "Statut": "✅" if score is not None else "❌",
                })

            df = pd.DataFrame(data)
            df = df.sort_values("#", ascending=False)

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Sélecteur pour voir le code
            st.markdown("#### 🔍 Voir le code d'un sample")
            sample_nums = [s.get("sample_order", 0) for s in samples]
            selected = st.selectbox("Sélectionner un sample", sorted(sample_nums, reverse=True))

            if selected is not None:
                for sample in samples:
                    if sample.get("sample_order") == selected:
                        st.code(sample.get("function", "# Pas de code"), language="python")
                        if sample.get("score"):
                            st.caption(f"Score: {sample.get('score'):.4f}")
                        break
        else:
            st.info("Aucun sample disponible.")

    # --- Onglet 4: Statistiques ---
    with tab4:
        if summary:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ⏱️ Temps d'exécution")
                sample_time = summary.get("total_sample_time", 0)
                eval_time = summary.get("total_evaluate_time", 0)

                time_df = pd.DataFrame({
                    "Étape": ["Sampling (LLM)", "Évaluation"],
                    "Temps (s)": [sample_time, eval_time]
                })
                st.bar_chart(time_df.set_index("Étape"))

            with col2:
                st.markdown("#### 📊 Répartition des samples")
                valid = summary.get("valid_samples", 0)
                failed = summary.get("failed_samples", 0)

                status_df = pd.DataFrame({
                    "Statut": ["Valides", "Échouées"],
                    "Nombre": [valid, failed]
                })
                st.bar_chart(status_df.set_index("Statut"))

            # Distribution des scores
            if all_scores:
                st.markdown("#### 📈 Distribution des scores")
                scores = [s["score"] for s in all_scores if s.get("score") is not None]

                if scores:
                    score_stats = summary.get("score_stats", {})

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max", f"{score_stats.get('max', max(scores)):.4f}")
                    with col2:
                        st.metric("Min", f"{score_stats.get('min', min(scores)):.4f}")
                    with col3:
                        mean_score = score_stats.get('mean', sum(scores)/len(scores))
                        st.metric("Moyenne", f"{mean_score:.4f}")
        else:
            st.info("Statistiques non disponibles.")


def _load_json(path: Path) -> Optional[Dict]:
    """Charge un fichier JSON."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


# =============================================================================
# Étape 5: AutoML
# =============================================================================

def render_automl_step():
    """Étape AutoML."""
    st.markdown("## 🤖 Étape 6: AutoML")

    with card_block():
        st.markdown("""
        AutoML va entraîner plusieurs modèles et sélectionner le meilleur.

        **Frameworks disponibles:**
        - AutoGluon (recommandé)
        - H2O AutoML
        - FLAML
        """)

        framework = st.selectbox(
            "Framework AutoML",
            ["autogluon", "h2o", "flaml"],
            index=0,
        )

        time_limit = st.slider(
            "Temps limite (minutes)",
            min_value=1,
            max_value=60,
            value=10,
        )

    if st.session_state.automl_results is None:
        if st.button("🚀 Lancer AutoML", type="primary", use_container_width=True):
            # ===== CHECKPOINT 0: Bouton cliqué =====
            print("\n" + "#"*60)
            print("[CHECKPOINT 0] BOUTON 'Lancer AutoML' CLIQUÉ!")
            print(f"  - Framework sélectionné: {framework}")
            print(f"  - Time limit: {time_limit} minutes")
            print(f"  - automl_results actuel: {st.session_state.automl_results}")
            print("#"*60 + "\n")
            st.write("🔍 Debug: Bouton cliqué, lancement en cours...")
            run_automl(framework, time_limit * 60)
    else:
        render_automl_results()

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Retour", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()
    with col2:
        if st.session_state.automl_results is not None:
            if st.button("Voir les résultats finaux →", type="primary", use_container_width=True):
                st.session_state.current_step = 6
                st.rerun()


def run_automl(framework: str, time_limit: int):
    """Lance AutoML."""
    import traceback
    from pathlib import Path
    import pandas as pd

    # ===== CHECKPOINT 1: Début =====
    print("\n" + "="*60)
    print("[CHECKPOINT 1] run_automl() APPELÉ")
    print(f"  - Framework: {framework}")
    print(f"  - Time limit: {time_limit} secondes")
    print("="*60)

    st.info(f"🔄 Lancement de {framework} (limite: {time_limit//60} min)...")

    progress = st.progress(0, text="Préparation des données...")

    try:
        # ===== CHECKPOINT 2: Import =====
        print("\n[CHECKPOINT 2] Tentative d'import de run_automl...")

        # Forcer le rechargement du module (évite les problèmes de cache)
        import importlib
        import src.automl.runner as runner_module
        importlib.reload(runner_module)
        automl_run = runner_module.run_automl

        print("[CHECKPOINT 2] ✅ Import réussi!")

        # ===== CHECKPOINT 2b: Vérifier si dataset FE existe =====
        df_to_use = st.session_state.df_train
        fe_csv_path = Path("outputs") / st.session_state.project_name / "feature_engineering" / "dataset_fe" / "train_fe.csv"

        if fe_csv_path.exists():
            print(f"\n[CHECKPOINT 2b] ✅ Dataset FE trouvé: {fe_csv_path}")
            df_to_use = pd.read_csv(fe_csv_path)
            print(f"  - Shape original: {st.session_state.df_train.shape}")
            print(f"  - Shape FE: {df_to_use.shape}")
            st.info(f"📊 Utilisation du dataset transformé ({df_to_use.shape[1]} features)")
        else:
            print(f"\n[CHECKPOINT 2b] ℹ️ Pas de dataset FE - utilisation des données originales")

        # ===== CHECKPOINT 3: Vérification données =====
        print("\n[CHECKPOINT 3] Vérification des données...")
        print(f"  - df_train shape: {df_to_use.shape}")
        print(f"  - target_col: {st.session_state.target_col}")
        print(f"  - project_name: {st.session_state.project_name}")

        progress.progress(20, text="Entraînement des modèles...")

        # ===== CHECKPOINT 4: Lancement =====
        print("\n[CHECKPOINT 4] Lancement de automl_run()...")
        results = automl_run(
            df_train=df_to_use,
            target_col=st.session_state.target_col,
            framework=framework,
            time_limit=time_limit,
            project_name=st.session_state.project_name,
        )

        # ===== CHECKPOINT 5: Résultats =====
        print("\n[CHECKPOINT 5] ✅ AutoML terminé!")
        print(f"  - Résultats: {results}")

        progress.progress(100, text="Terminé!")
        st.session_state.automl_results = results

        st.success("✅ AutoML terminé!")
        st.rerun()

    except ImportError as e:
        # ===== CHECKPOINT ERROR: Import =====
        print("\n[CHECKPOINT ERROR] ❌ ImportError!")
        print(f"  - Erreur: {e}")
        print(f"  - Traceback:\n{traceback.format_exc()}")

        progress.empty()
        st.warning(f"⚠️ Module AutoML non disponible: {e}")
        st.code(traceback.format_exc())

        # Résultats simulés pour la démo
        st.session_state.automl_results = {
            "best_model": "XGBoost (simulé)",
            "best_score": 0.85,
            "framework": framework,
            "models_tested": 10,
            "error": str(e),
        }
        st.rerun()

    except Exception as e:
        # ===== CHECKPOINT ERROR: Autre =====
        print("\n[CHECKPOINT ERROR] ❌ Exception!")
        print(f"  - Type: {type(e).__name__}")
        print(f"  - Erreur: {e}")
        print(f"  - Traceback:\n{traceback.format_exc()}")

        progress.empty()
        st.error(f"❌ Erreur AutoML: {e}")
        st.code(traceback.format_exc())


def render_automl_results():
    """Affiche les résultats AutoML."""
    results = st.session_state.automl_results

    with card_block():
        st.markdown("### ✅ AutoML terminé")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Meilleur modèle", results.get("best_model", "N/A"))
        with col2:
            score = results.get('best_score')
            st.metric("Score", f"{score:.4f}" if score is not None else "N/A")
        with col3:
            st.metric("Modèles testés", results.get("models_tested", 0))


# =============================================================================
# Étape 6: Résultats finaux
# =============================================================================

def render_results_step():
    """Affiche les résultats finaux du pipeline."""
    st.markdown("## 📈 Résultats du Pipeline")

    st.balloons()

    with card_block():
        st.markdown("### 🎉 Pipeline terminé avec succès!")

        st.markdown(f"""
        **Projet:** {st.session_state.project_name}

        **Résumé:**
        - Dataset: {len(st.session_state.df_train)} lignes
        - Cible: `{st.session_state.target_col}`
        - Type: {"Régression" if st.session_state.is_regression else "Classification"}
        """)

    # Résultats par étape
    col1, col2 = st.columns(2)

    with col1:
        with card_block():
            st.markdown("### 📊 Analyse")
            if st.session_state.stats_report:
                n_cols = len(st.session_state.stats_report.get("columns", {}))
                st.metric("Features analysées", n_cols)

    with col2:
        with card_block():
            st.markdown("### 🔧 Feature Engineering")
            if st.session_state.fe_results:
                st.markdown(f"Dossier: `{st.session_state.fe_results.get('results_dir', 'N/A')}`")

    # AutoML
    with card_block():
        st.markdown("### 🤖 AutoML")
        if st.session_state.automl_results:
            results = st.session_state.automl_results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Meilleur modèle", results.get("best_model", "N/A"))
            with col2:
                st.metric("Score", f"{results.get('best_score', 0):.4f}")
            with col3:
                st.metric("Framework", results.get("framework", "N/A"))

    # Bouton recommencer
    st.markdown("---")
    if st.button("🔄 Nouveau projet", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# =============================================================================
# Main App
# =============================================================================

def main():
    """Point d'entrée principal."""
    render_header()
    render_progress_bar()
    render_sidebar()

    # Router vers l'étape courante
    step = st.session_state.current_step

    if step == 0:
        render_upload_step()
    elif step == 1:
        render_config_step()
    elif step == 2:
        render_analyse_step()
    elif step == 3:
        render_chat_step()
    elif step == 4:
        render_fe_step()
    elif step == 5:
        render_automl_step()
    elif step == 6:
        render_results_step()


if __name__ == "__main__":
    main()
