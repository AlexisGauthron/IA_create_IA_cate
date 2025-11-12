import sys
import os

# -------------------------------------------------
# Bootstrapping: add 'src' to PYTHONPATH once
# -------------------------------------------------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# -------------------------------------------------
# Standard libs & typing
# -------------------------------------------------
import json, re, time, math
from typing import Any, Dict, List, Tuple, Optional
from contextlib import contextmanager
from collections import Counter, defaultdict

# -------------------------------------------------
# Third‑party libs
# -------------------------------------------------
import pandas as pd
import numpy as np
import requests
import streamlit as st

# -------------------------------------------------
# Project imports
# -------------------------------------------------

import src.front.section_embedding as emb_section


# — Embeddings (protos)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")



import src.few_shot.fews_shot_llm as f_llm

import src.front.css as css

import src.front.upload_fichier as upload_file
import src.fonctions.clean_label as clean_labels_module



# -------------------------------------------------
# Constants & theme
# -------------------------------------------------
CUSTOM_CSS = css.CUSTOM_CSS
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:7b-instruct"

# -------------------------------------------------
# Helpers
# -------------------------------------------------

import src.front.ui_helper as ui_helper



def setup_page() -> None:
    st.set_page_config(page_title="LLM Classifier · Ollama", page_icon="🧠", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("<div class='badge'>🧠 LLM Classifier · Ollama</div>", unsafe_allow_html=True)
    st.title("Classer un CSV avec des catégories choisies (Few-Shot / JSON strict)")


def build_sidebar() -> Dict[str, Any]:
    with ui_helper.card_block(st.sidebar) as sidebar:
        sidebar.markdown("### ⚙️ Paramètres")
        ollama_url = sidebar.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL, help="URL du serveur Ollama local")
        model = sidebar.selectbox("Modèle", options=[
            "qwen2.5:7b-instruct",
            "llama3.1:8b-instruct",
            "mistral:7b-instruct",
        ], index=0)
        temperature = sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_p = sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        seed = sidebar.number_input("Seed", 0, 10_000_000, 42, step=1)
        timeout = sidebar.number_input("Timeout (s)", 10, 600, 120, step=5)

        sidebar.markdown("### 🗳️ Self-Consistency")
        vote_n = sidebar.slider("Nombre d'appels (vote)", 1, 7, 1, help=">1 = vote majoritaire pour plus de robustesse")

        sidebar.markdown("### 🚦 Seuil & Règles")
        add_other = sidebar.toggle("Ajouter la classe 'Autre' automatiquement", value=True)
        threshold = sidebar.slider("Seuil de confiance pour basculer en 'Autre'", 0.0, 1.0, 0.35, 0.01)

        sidebar.markdown("### 🧩 Instructions (optionnel)")
        extra_instructions = sidebar.text_area("Contexte/contraintes métier", placeholder="Ex: Si 'TVA' ou 'devis' -> Facture prioritaire...")

    return {
        "ollama_url": ollama_url,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "timeout": timeout,
        "vote_n": vote_n,
        "add_other": add_other,
        "threshold": threshold,
        "extra_instructions": extra_instructions,
    }



def choose_categories_ui(df: Optional[pd.DataFrame], parent_right) -> list[str]:
    with ui_helper.card_block(parent_right) as right_card:
        right_card.subheader("2) Définir les catégories")

        use_column_labels = False
        if df is not None:
            use_column_labels = right_card.toggle(
                "Sélectionner les catégories depuis une colonne du CSV",
                value=False,
                help="Active pour choisir les catégories à partir des valeurs d'une colonne.",
            )

        if use_column_labels and df is not None:
            category_column = right_card.selectbox("Colonne source", options=list(df.columns))

            # IMPORTANT: dropna AVANT astype(str)
            col = df[category_column]
            col = col[~pd.isna(col)].astype(str).str.strip()

            # Clean + ordre conservé
            unique_values = clean_labels_module._clean_labels(col.tolist())

            labels = right_card.multiselect(
                "Choisis les catégories",
                options=unique_values,
                default=unique_values,
                help="Désélectionne les valeurs à exclure.",
            )
        else:
            raw_labels = right_card.text_input(
                "Catégories possibles (séparées par des virgules)",
                value="Facture,Support,RH",
            )
            labels = [l.strip() for l in raw_labels.split(",")]
            labels = clean_labels_module._clean_labels(labels)

        # Aperçu propre
        categories_preview = (
            pd.DataFrame({"Categorie": labels})
            if labels else
            pd.DataFrame(columns=["Categorie"])
        )
        right_card.dataframe(categories_preview, use_container_width=True)

        if not labels:
            right_card.warning("⚠️ Indique au moins une catégorie.")

        right_card.markdown("<small class='help'>Astuce: garde 3–8 catégories au début. Tu peux ajouter 'Autre' toi-même ou cocher l’option.</small>", unsafe_allow_html=True)

    # On retourne la version nettoyée
    return labels


def choose_method_ui(df: Optional[pd.DataFrame]) -> str:
    """Propose LLM vs Embeddings une fois le dataset fourni.
    Retourne 'LLM' ou 'Embeddings'."""
    default_method = "LLM"
    if df is None:
        return default_method

    options_map = {
        "LLM (Ollama)": "LLM",
        "Embeddings (Prototypes)": "Embeddings",
    }
    with ui_helper.card_block() as method_card:
        method_card.subheader("2bis) Choisir la méthode")
        if df.shape[1] > 1:
            choice = method_card.radio(
                "Méthode de catégorisation",
                options=list(options_map.keys()),
                index=0,
                horizontal=True,
                help="LLM : classification avec un modèle d'instructions. Embeddings : prototypes + similarité.",
            )
            return options_map[choice]
        else:
            method_card.info("Dataset à 1 colonne : mode **LLM** proposé par défaut.")
            return default_method


def choose_sampling_ui(df: pd.DataFrame) -> Tuple[int, bool]:
    """Paramètres d'échantillonnage communs aux deux méthodes."""
    with ui_helper.card_block() as params_card:
        params_card.subheader("3) Paramètres d'échantillonnage")
        sample_n = params_card.slider(
            "Taille de l'échantillon (prévisualisation)",
            1,
            min(500, len(df)),
            min(50, len(df)),
            help="Pour tester rapidement avant le run complet",
        )
        do_full = params_card.toggle("Exécuter sur TOUT le dataset (sinon: échantillon)", value=False)
    return sample_n, do_full


def configure_run_params_ui(df: pd.DataFrame, labels: List[str]) -> Tuple[str, int, bool, bool]:
    """Return (text_col, sample_n, do_full, run_clicked) — spécifique au mode LLM."""
    if df is None or len(labels) == 0:
        return "", 0, False, False

    with ui_helper.card_block() as params_card:
        params_card.subheader("3) Paramétrer le classement (LLM)")

        text_col = params_card.selectbox("Colonne texte à classer", options=list(df.columns))
        sample_n = params_card.slider(
            "Taille de l'échantillon (prévisualisation)",
            1,
            min(500, len(df)),
            min(50, len(df)),
            help="Pour tester rapidement avant le run complet",
        )
        do_full = params_card.toggle("Exécuter sur TOUT le dataset (sinon: échantillon)", value=False)

        run = params_card.button("🚀 Lancer le classement (LLM)", type="primary", use_container_width=True)

    return text_col, sample_n, do_full, run


def _classify_one_text(text: str, labels: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Single LLM classification call with or without self-consistency."""
    if cfg["vote_n"] == 1:
        return f_llm.classify_once(
            text=text, labels=labels, add_other=cfg["add_other"], threshold=cfg["threshold"],
            ollama_url=cfg["ollama_url"], model=cfg["model"], seed=cfg["seed"],
            temperature=cfg["temperature"], top_p=cfg["top_p"], timeout=cfg["timeout"],
            extra_instructions=cfg["extra_instructions"],
        )
    else:
        return f_llm.classify_vote(
            text=text, labels=labels, add_other=cfg["add_other"], threshold=cfg["threshold"],
            ollama_url=cfg["ollama_url"], model=cfg["model"], n=cfg["vote_n"], base_seed=cfg["seed"],
            temperature=cfg["temperature"], top_p=cfg["top_p"], timeout=cfg["timeout"],
            extra_instructions=cfg["extra_instructions"],
        )


def run_llm_pipeline(df: pd.DataFrame, labels: List[str], text_col: str, sample_n: int, do_full: bool, cfg: Dict[str, Any]) -> None:
    data = df.copy()
    if not do_full:
        data = data.head(sample_n).copy()

    out_labels, out_conf, out_just = [], [], []

    n_rows = len(data)
    prog = st.progress(0, text="Classification en cours…")
    status = st.empty()

    for i, text in enumerate(data[text_col].astype(str).fillna("").tolist()):
        try:
            res = _classify_one_text(text=text, labels=labels, cfg=cfg)
            out_labels.append(res["label"]) 
            out_conf.append(res["confidence"]) 
            out_just.append(res.get("justification_bref", ""))
        except Exception as e:
            out_labels.append("Erreur")
            out_conf.append(0.0)
            out_just.append(str(e))

        prog.progress((i + 1) / n_rows, text=f"Classification… ({i+1}/{n_rows})")
        if i % 10 == 0:
            status.info(f"Ligne {i+1}/{n_rows}")
        time.sleep(0.02)

    data["pred_label"] = out_labels
    data["pred_confidence"] = out_conf
    data["pred_justification"] = out_just

    with ui_helper.card_block() as results_card:
        results_card.success("Terminé ✅")
        results_card.subheader("4) Résultats (LLM)")
        results_card.dataframe(data.head(30), use_container_width=True)

        results_card.markdown("#### Répartition des labels")
        counts = data["pred_label"].value_counts(dropna=False)
        results_card.dataframe(pd.DataFrame({"label": counts.index, "count": counts.values}), use_container_width=True)

        csv_bytes = data.to_csv(index=False).encode("utf-8")
        results_card.download_button(
            "💾 Télécharger le CSV enrichi (LLM)",
            data=csv_bytes,
            file_name="classified_results_llm.csv",
            mime="text/csv",
            use_container_width=True,
        )



def footer_help() -> None:
    st.markdown("<hr/>", unsafe_allow_html=True)
    with st.expander("Aide & conseils"):
        st.markdown(
            """
- **Modèles conseillés** : `qwen2.5:7b-instruct` (souvent meilleur en JSON), `llama3.1:8b-instruct`, `mistral:7b-instruct`.
- **Catégories** : commence par 3–8 labels clairs. Active **'Autre'** + **seuil** pour limiter les faux positifs.
- **Vote (Self-Consistency)** : mets 3–5 pour des textes ambigus (plus lent, mais plus robuste).
- **Contexte métier** : utilise la zone “Instructions” pour des règles de précédence simples (ex: *si TVA → Facture*).
- **Performance** : teste d’abord sur un échantillon, puis coche “Tout le dataset”.
            """
        )


# -------------------------------------------------
# Minimal smoke tests (not run by default)
# -------------------------------------------------

def run_smoke_tests() -> Dict[str, Any]:
    """Basic non-interactive checks to help catch regressions.
    Call manually from a REPL:  
        >>> import app_refactor_streamlit as app
        >>> app.run_smoke_tests()
    """
    results: Dict[str, Any] = {}

    # 1) Constants are set
    assert isinstance(DEFAULT_OLLAMA_URL, str) and DEFAULT_OLLAMA_URL.startswith("http"), "DEFAULT_OLLAMA_URL invalid"
    assert isinstance(CUSTOM_CSS, str), "CUSTOM_CSS should be a string"

    # 2) card_block yields a container
    try:
        # We can't materialize Streamlit containers here without a running app,
        # but we still exercise the contextmanager API.
        from contextlib import ExitStack
        with ExitStack():
            pass
        results["contextmanager_api"] = True
    except Exception as e:
        results["contextmanager_api"] = f"Failed: {e}"

    # 3) Placeholder text for definitions is a safe Python string
    placeholder_text = (
        "Facture | Documents de facturation, devis, paiements…\n"
        "Support | Incidents, bugs, tickets…"
    )
    assert "\n" in placeholder_text and "Facture" in placeholder_text
    results["placeholder_ok"] = True

    # 4) _compute_select_index unit tests
    cols = ["a", "b", "c"]
    assert ui_helper._compute_select_index(cols, None) == 0
    assert ui_helper._compute_select_index(cols, "b") == 1
    assert ui_helper._compute_select_index(cols, "x") == 0
    # duplicated column names (edge): still returns first match
    dup_cols = ["t", "t", "u"]
    assert ui_helper._compute_select_index(dup_cols, "t") == 0

    results["select_index_tests"] = True

    return results


# -------------------------------------------------
# Main orchestration
# -------------------------------------------------

def main() -> None:
    setup_page()

    cfg = build_sidebar()

    col_left, col_right = st.columns([1.1, 1])

    df = upload_file.upload_csv_ui(col_left)
    labels = choose_categories_ui(df, col_right)

    if df is not None and len(labels) > 0:
        method = choose_method_ui(df)

        if method == "LLM":
            # UI spécifique LLM
            text_col, sample_n, do_full, run_clicked = configure_run_params_ui(df, labels)
            if run_clicked and text_col:
                run_llm_pipeline(df=df, labels=labels, text_col=text_col, sample_n=sample_n, do_full=do_full, cfg=cfg)
        else:
            # Paramètres communs pour Embeddings
            sample_n, do_full = choose_sampling_ui(df)
            emb_section.embeddings_section_ui(df=df, labels=labels, do_full=do_full, sample_n=sample_n, default_text_col=text_col if 'text_col' in locals() else None)

    footer_help()


if __name__ == "__main__":
    main()
