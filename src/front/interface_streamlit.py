import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# app.py
import json, re, time, math
from typing import Any, Dict, List
from contextlib import contextmanager
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import requests
import streamlit as st


import src.categorisation_echantillion.fews_shot_llm as f_llm
import src.front.css as css

# — Embeddings (protos)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


from src.categorisation_echantillion.fews_shot_embedding import (
    load_model as emb_load_model,
    build_prototypes as emb_build_prototypes,
    classify_one as emb_classify_one,
    classify_one_multi as emb_classify_one_multi,
    calibrate_threshold as emb_calibrate_threshold,
)


CUSTOM_CSS = css.CUSTOM_CSS


# --------------------------
# Page / thème
# --------------------------
st.set_page_config(page_title="LLM Classifier · Ollama", page_icon="🧠", layout="wide")



st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("<div class='badge'>🧠 LLM Classifier · Ollama</div>", unsafe_allow_html=True)
st.title("Classer un CSV avec des catégories choisies (Few-Shot / JSON strict)")

# --------------------------
# Config par défaut
# --------------------------
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:7b-instruct"


@contextmanager
def card_block(parent=st, border=True):
    c = parent.container(border=border)   # <-- le wrapper réel des widgets
    with c:
        # Optionnel: déco interne (restera bien DANS la card grâce au position:relative ci-dessus)
        # st.markdown('<div class="card-ribbon" style="background: linear-gradient(90deg,#00f260,#0575e6)"></div>',
        #             unsafe_allow_html=True)
        yield c


# --------------------------
# Sidebar (contrôles)
# --------------------------
with card_block(st.sidebar) as sidebar:
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

# --------------------------
# Input zone
# --------------------------
col_left, col_right = st.columns([1.1, 1])

with card_block(col_left) as left_card:
    left_card.subheader("1) Charger votre CSV")
    uploaded = left_card.file_uploader("CSV à classer", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, sep=";")
        left_card.caption(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        left_card.dataframe(df.head(12), use_container_width=True)
    else:
        df = None
        left_card.info("Chargez un fichier CSV pour commencer.")

labels: List[str] = []
use_column_labels = False

with card_block(col_right) as right_card:
    right_card.subheader("2) Définir les catégories")

    if df is not None:
        use_column_labels = right_card.toggle(
            "Sélectionner les catégories depuis une colonne du CSV",
            value=False,
            help="Active pour choisir les catégories à partir des valeurs d'une colonne.",
        )

    if use_column_labels and df is not None:
        category_column = right_card.selectbox("Colonne source", options=list(df.columns))
        unique_values = (
            df[category_column]
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        unique_values = sorted(set(unique_values))
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
        labels = [l.strip() for l in raw_labels.split(",") if l.strip()]

    categories_preview = pd.DataFrame({"Categorie": labels}) if labels else pd.DataFrame(columns=["Categorie"])
    right_card.dataframe(categories_preview, use_container_width=True)

    if not labels:
        right_card.warning("⚠️ Indique au moins une catégorie.")

    right_card.markdown("<small class='help'>Astuce: garde 3–8 catégories au début. Tu peux ajouter 'Autre' toi-même ou cocher l’option.</small>", unsafe_allow_html=True)

# --------------------------
# Sélection de colonnes + options de batch
# --------------------------
if df is not None and len(labels) > 0:
    with card_block() as params_card:
        params_card.subheader("3) Paramétrer le classement")

        text_col = params_card.selectbox("Colonne texte à classer", options=list(df.columns))
        sample_n = params_card.slider(
            "Taille de l'échantillon (prévisualisation)",
            1,
            min(500, len(df)),
            min(50, len(df)),
            help="Pour tester rapidement avant le run complet",
        )
        do_full = params_card.toggle("Exécuter sur TOUT le dataset (sinon: échantillon)", value=False)

        run = params_card.button("🚀 Lancer le classement", type="primary", use_container_width=True)

    # ----------------------
    # Exécution
    # ----------------------
    if run:
        data = df.copy()
        if not do_full:
            data = data.head(sample_n).copy()

        out_labels = []
        out_conf = []
        out_just = []

        n_rows = len(data)
        prog = st.progress(0, text="Classification en cours…")
        status = st.empty()

        for i, text in enumerate(data[text_col].astype(str).fillna("").tolist()):
            try:
                if vote_n == 1:
                    res = f_llm.classify_once(
                        text=text, labels=labels, add_other=add_other, threshold=threshold,
                        ollama_url=ollama_url, model=model, seed=seed, temperature=temperature,
                        top_p=top_p, timeout=timeout, extra_instructions=extra_instructions
                    )
                else:
                    res = f_llm.classify_vote(
                        text=text, labels=labels, add_other=add_other, threshold=threshold,
                        ollama_url=ollama_url, model=model, n=vote_n, base_seed=seed,
                        temperature=temperature, top_p=top_p, timeout=timeout, extra_instructions=extra_instructions
                    )
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

            # petite temporisation pour éviter de saturer (ajuste si besoin)
            time.sleep(0.02)

        data["pred_label"] = out_labels
        data["pred_confidence"] = out_conf
        data["pred_justification"] = out_just

        with card_block() as results_card:
            results_card.success("Terminé ✅")
            results_card.subheader("4) Résultats")
            results_card.dataframe(data.head(30), use_container_width=True)

            results_card.markdown("#### Répartition des labels")
            counts = data["pred_label"].value_counts(dropna=False)
            results_card.dataframe(pd.DataFrame({"label": counts.index, "count": counts.values}), use_container_width=True)

            csv_bytes = data.to_csv(index=False).encode("utf-8")
            results_card.download_button(
                "💾 Télécharger le CSV enrichi",
                data=csv_bytes,
                file_name="classified_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================
#  EMBEDDINGS (PROTOTYPES)
# ============================
if df is not None and len(labels) > 0:
    with card_block() as emb_card:
        emb_card.subheader("🔎 Embeddings (prototypes)")

        # Choix du modèle d'embeddings
        emb_model_name = emb_card.selectbox(
            "Modèle d'embeddings",
            options=[
                "intfloat/multilingual-e5-base",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "BAAI/bge-m3",
            ],
            index=0,
            help="E5 conseillé (ajoute automatiquement les préfixes query:/passage:)."
        )

        # Source des prototypes
        shot_source = emb_card.radio(
            "Prototypes depuis…",
            options=["CSV labellisé", "Saisie manuelle"],
            index=0,
            help="CSV : utilise une colonne label existante pour constituer les exemples."
        )

        alpha = emb_card.slider("Poids de la définition (alpha)", 0.0, 0.8, 0.3, 0.05,
                                help="0 = off ; mélange le prototype avec la définition si fournie.")
        allow_other = emb_card.toggle("Activer le rejet 'Autre'", value=True)
        use_multi = emb_card.toggle("Mode multi-label", value=False)

        # --------- Constitution des shots (exemples) ---------
        shots: Dict[str, List[str]] = {}
        label_defs: Dict[str, str] = {}

        if shot_source == "CSV labellisé":
            # Colonne texte et colonne label (pour les prototypes)
            emb_text_col = emb_card.selectbox("Colonne texte (protos & à classer)", options=list(df.columns), index=list(df.columns).index(text_col) if 'text_col' in locals() else 0)
            label_col = emb_card.selectbox("Colonne label (pour construire les prototypes)", options=list(df.columns))
            k_per_label = emb_card.slider("Exemples par label (pour proto)", 1, 50, 5, help="On prélève les k premières occurrences par label.")
            # Option : définitions des labels (facultatif)
            with emb_card.expander("Définitions des labels (optionnel)"):
                emb_card.markdown("Format: **Label | définition** par ligne")
                defs_raw = emb_card.text_area("Définitions", value="", height=120, placeholder="Facture | Documents de facturation, devis, paiements…\nSupport | Incidents, bugs, tickets…")
                if defs_raw.strip():
                    for line in defs_raw.splitlines():
                        if "|" in line:
                            k, v = line.split("|", 1)
                            label_defs[k.strip()] = v.strip()

            # Construire shots depuis le DF
            if label_col and emb_text_col:
                tmp = df[[emb_text_col, label_col]].dropna()
                # garder seulement les labels sélectionnés (si l'utilisateur a filtré via la carte 2)
                tmp = tmp[tmp[label_col].astype(str).isin(labels)]
                for lbl in labels:
                    exs = tmp[tmp[label_col].astype(str) == lbl][emb_text_col].astype(str).head(k_per_label).tolist()
                    if exs:
                        shots[lbl] = exs

        else:
            # Saisie manuelle : une zone par label
            with emb_card.expander("Saisir des exemples (un par ligne) + définition optionnelle"):
                for lbl in labels:
                    col1, col2 = emb_card.columns([1, 1])
                    with col1:
                        ex_raw = st.text_area(f"Exemples — {lbl}", value="", height=100, key=f"shots_{lbl}")
                        examples = [l.strip() for l in ex_raw.splitlines() if l.strip()]
                        if examples:
                            shots[lbl] = examples
                    with col2:
                        d_raw = st.text_area(f"Définition — {lbl} (optionnel)", value="", height=100, key=f"def_{lbl}")
                        if d_raw.strip():
                            label_defs[lbl] = d_raw.strip()
        
        # Aperçu des shots
        with emb_card.expander("Aperçu des prototypes (shots)"):
            if shots:
                preview_rows = []
                for k, vs in shots.items():
                    preview_rows.append({"label": k, "n_exemples": len(vs), "exemple_1": (vs[0] if vs else "")})
                emb_card.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
            else:
                emb_card.info("Aucun exemple disponible pour construire les prototypes.")

        # --------- Calibration des seuils ---------
        colA, colB = emb_card.columns([1, 1])
        with colA:
            calib_btn = st.button("🧪 Calibrer (leave-one-out)", use_container_width=True)
        with colB:
            run_embed_btn = st.button("🚀 Lancer (Embeddings)", type="primary", use_container_width=True)

        # Valeurs par défaut si pas de calibration
        thr_default, mar_default = 0.35, 0.05
        thr_val, mar_val = thr_default, mar_default

        if calib_btn:
            try:
                emb_load_model(emb_model_name)
                thr_val, mar_val = emb_calibrate_threshold(shots, label_defs if label_defs else None, alpha=alpha)
                emb_card.success(f"Seuil calibré ≈ {thr_val:.2f} | Marge ≈ {mar_val:.2f}")
            except Exception as e:
                emb_card.error(f"Calibration impossible : {e}")
        else:
            # montre les valeurs actuelles (par défaut tant qu'on n'a pas calibré)
            emb_card.caption(f"Seuil par défaut: {thr_val} | Marge par défaut: {mar_val}")

        # --------- Exécution classification embeddings ---------
        if run_embed_btn:
            if not shots:
                emb_card.error("Aucun prototype. Fournis des exemples (CSV labellisé ou saisie manuelle).")
            else:
                try:
                    emb_load_model(emb_model_name)
                    protos = emb_build_prototypes(shots, label_defs if label_defs else None, alpha=alpha)

                    # Quelle colonne on classe ? (réutilise text_col si présent)
                    classify_col = emb_text_col if shot_source == "CSV labellisé" else (
                        text_col if 'text_col' in locals() else st.selectbox("Colonne texte à classer", options=list(df.columns))
                    )

                    data2 = df.copy()
                    if not do_full:
                        data2 = data2.head(sample_n).copy()

                    n_rows2 = len(data2)
                    prog2 = st.progress(0, text="Embeddings : classification en cours…")
                    res_labels, res_conf, res_extra = [], [], []

                    for i, txt in enumerate(data2[classify_col].astype(str).fillna("").tolist()):
                        if use_multi:
                            r = emb_classify_one_multi(txt, protos, per_label_threshold=max(0.4, thr_val))
                            res_labels.append(", ".join(r["labels"]))
                            # on affiche la meilleure sim comme "confidence" indicative
                            best = max((r["sims"].values() or [0.0]))
                            res_conf.append(float(best))
                            res_extra.append(r["sims"])
                        else:
                            r = emb_classify_one(txt, protos, threshold=thr_val, margin=mar_val, allow_other=allow_other)
                            res_labels.append(r["label"])
                            res_conf.append(r["confidence"])
                            res_extra.append(r["sims"])

                        prog2.progress((i + 1) / n_rows2, text=f"Embeddings… ({i+1}/{n_rows2})")

                    data2["emb_label"] = res_labels
                    data2["emb_confidence"] = res_conf

                    with card_block() as out_card:
                        out_card.success("Terminé (Embeddings) ✅")
                        out_card.dataframe(data2.head(30), use_container_width=True)

                        # répartition
                        if not use_multi:
                            counts2 = data2["emb_label"].value_counts(dropna=False)
                            out_card.markdown("#### Répartition des labels (Embeddings)")
                            out_card.dataframe(pd.DataFrame({"label": counts2.index, "count": counts2.values}), use_container_width=True)

                        # export
                        csv2 = data2.to_csv(index=False).encode("utf-8")
                        out_card.download_button(
                            "💾 Télécharger CSV (Embeddings)",
                            data=csv2,
                            file_name="classified_results_embeddings.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                except Exception as e:
                    emb_card.error(f"Erreur embeddings : {e}")


# --------------------------
# Pied / aide
# --------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
with st.expander("Aide & conseils"):
    st.markdown("""
- **Modèles conseillés** : `qwen2.5:7b-instruct` (souvent meilleur en JSON), `llama3.1:8b-instruct`, `mistral:7b-instruct`.
- **Catégories** : commence par 3–8 labels clairs. Active **'Autre'** + **seuil** pour limiter les faux positifs.
- **Vote (Self-Consistency)** : mets 3–5 pour des textes ambigus (plus lent, mais plus robuste).
- **Contexte métier** : utilise la zone “Instructions” pour des règles de précédence simples (ex: *si TVA → Facture*).
- **Performance** : teste d’abord sur un échantillon, puis coche “Tout le dataset”.
""")
