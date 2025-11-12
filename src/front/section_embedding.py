import sys
import os

# -------------------------------------------------
# Bootstrapping: add 'src' to PYTHONPATH once
# -------------------------------------------------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# — Embeddings (protos)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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


import src.few_shot.prototypical.few_shot as few_shot_emb
import src.few_shot.prototypical.classify as emb_classify


# -------------------------------------------------
# Helpers
# -------------------------------------------------
import src.front.css as css
import src.front.ui_helper as ui_helper

# -------------------------------------------------
# Constants & theme
# -------------------------------------------------
CUSTOM_CSS = css.CUSTOM_CSS
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:7b-instruct"


import streamlit as st
import pandas as pd
from typing import Dict, List, Optional

def embeddings_section_ui(
    df: pd.DataFrame,
    labels: List[str],
    do_full: bool,
    sample_n: int,
    default_text_col: Optional[str]
) -> None:
    if df is None or len(labels) == 0:
        return

    # ---------- FIX: initialiser les clés Session State avant widgets ----------
    st.session_state.setdefault("defs_raw", "")
    st.session_state.setdefault("apply_defs_raw", False)
    st.session_state.setdefault("pending_defs_raw_lines", [])

    # Pour le mode manuel : on prépare def_*, shots_* et un buffer pour apply
    for _lbl in labels:
        st.session_state.setdefault(f"def_{_lbl}", "")
        st.session_state.setdefault(f"shots_{_lbl}", "")
    st.session_state.setdefault("apply_def_map", False)
    st.session_state.setdefault("pending_def_map", {})  # {label: def_str}

    # ---------- Si on a un pending à appliquer, le faire AVANT de créer widgets ----------
    if st.session_state.get("apply_defs_raw"):
        st.session_state["defs_raw"] = "\n".join(st.session_state.get("pending_defs_raw_lines", []))
        st.session_state["apply_defs_raw"] = False
        st.session_state["pending_defs_raw_lines"] = []

    if st.session_state.get("apply_def_map"):
        for k, v in st.session_state.get("pending_def_map", {}).items():
            st.session_state[f"def_{k}"] = v
        st.session_state["apply_def_map"] = False
        st.session_state["pending_def_map"] = {}

    # ---------------- UI principale ----------------
    with ui_helper.card_block() as emb_card:
        emb_card.subheader("🔎 Embeddings (prototypes)")

        # Modèle d'embeddings
        emb_model_name = emb_card.selectbox(
            "Modèle d'embeddings",
            options=[
                "intfloat/multilingual-e5-base",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "BAAI/bge-m3",
            ],
            index=0,
            help="E5 conseillé (ajoute automatiquement les préfixes query:/passage:).",
        )

        # Source des prototypes
        shot_source = emb_card.radio(
            "Prototypes depuis…",
            options=["CSV labellisé", "Saisie manuelle"],
            index=0,
            help="CSV : utilise une colonne label existante pour constituer les exemples.",
        )

        allow_other = emb_card.toggle("Activer le rejet 'Autre'", value=True)
        use_multi = emb_card.toggle("Mode multi-label", value=False)

        # --------- Constitution des shots / defs ---------
        shots: Dict[str, List[str]] = {}
        label_defs: Dict[str, str] = {}

        few_shot_embedding = few_shot_emb.FewShotExperiment(model_name=emb_model_name)  # initialise le modèle d'embeddings

        if shot_source == "CSV labellisé":
            emb_text_col = emb_card.selectbox(
                "Colonne texte (protos & à classer)",
                options=list(df.columns),
                index= ui_helper._compute_select_index(list(df.columns), default_text_col),
            )
            label_col = emb_card.selectbox(
                "Colonne label (pour construire les prototypes)",
                options=list(df.columns)
            )

            k_per_label = emb_card.slider(
                "Exemples par label (pour proto)", 1, 50, 5,
                help="On prélève les k premières occurrences par label."
            )

            with emb_card.expander("Définitions des labels (optionnel)"):
                emb_card.markdown("Format: **Label | définition** par ligne")
                # FIX: valeur prise depuis session_state (clé identique)
                emb_card.text_area(
                    "Définitions",
                    value=st.session_state["defs_raw"],
                    height=120,
                    key="defs_raw",
                    placeholder=(
                        "Facture | Documents de facturation, devis, paiements…\n"
                        "Support | Incidents, bugs, tickets…"
                    ),
                )

                # --- Bouton: Générer automatiquement ---
                if emb_card.button(
                    "✨ Générer automatiquement les définitions (Ollama)",
                    use_container_width=True,
                    key="btn_gen_defs_csv"
                ):
                    # 1) Construire des shots à partir du CSV courant
                    tmp = df[[emb_text_col, label_col]].dropna()
                    tmp = tmp[tmp[label_col].astype(str).isin(labels)]
                    shots_current: Dict[str, List[str]] = {}
                    for lbl in labels:
                        exs = (
                            tmp[tmp[label_col].astype(str) == lbl][emb_text_col]
                            .astype(str).head(k_per_label).tolist()
                        )
                        if exs:
                            shots_current[lbl] = exs


                    few_shot_embedding.set_shots(shots_current)


                    # 2) Appeler le LLM pour définitions concises
                    try:
                        # Assure-toi que get_def est importé, ex:
                        # from src.categorisation_echantillion import get_def as get_def
                        defs_map = few_shot_embedding.set_definitions(
                            allow_defs=True,
                            label_defs=None,
                            model_defs="mistral:7b-instruct",
                            max_terms_defs=10,
                            temperature_defs=0.0,
                        )
                    except Exception as e:
                        defs_map = {k: "" for k in shots_current}
                        emb_card.warning(f"LLM indisponible, fallback local utilisé. Détails: {e}")

                    # 3) Buffer + flag puis rerun (évite l'erreur Streamlit)
                    lines = [f"{k} | {v}" for k, v in defs_map.items()]
                    st.session_state["pending_defs_raw_lines"] = lines
                    st.session_state["apply_defs_raw"] = True
                    st.rerun()

                # Parsing du contenu actuel
                if st.session_state["defs_raw"].strip():
                    for line in st.session_state["defs_raw"].splitlines():
                        if "|" in line:
                            k, v = line.split("|", 1)
                            label_defs[k.strip()] = v.strip()

            # Shots depuis CSV
            tmp = df[[emb_text_col, label_col]].dropna()
            tmp = tmp[tmp[label_col].astype(str).isin(labels)]
            for lbl in labels:
                exs = (
                    tmp[tmp[label_col].astype(str) == lbl][emb_text_col]
                    .astype(str).head(k_per_label).tolist()
                )
                if exs:
                    shots[lbl] = exs

        else:
            with emb_card.expander("Saisir des exemples (un par ligne) + définition optionnelle"):
                for lbl in labels:
                    col1, col2 = emb_card.columns([1, 1])

                    with col1:
                        # FIX: utiliser session_state pour value + même clé
                        ex_raw = st.text_area(
                            f"Exemples — {lbl}",
                            value=st.session_state[f"shots_{lbl}"],
                            height=100,
                            key=f"shots_{lbl}",
                        )
                        examples = [l.strip() for l in ex_raw.splitlines() if l.strip()]
                        if examples:
                            shots[lbl] = examples

                    with col2:
                        d_raw = st.text_area(
                            f"Définition — {lbl} (optionnel)",
                            value=st.session_state[f"def_{lbl}"],
                            height=100,
                            key=f"def_{lbl}",
                        )
                        if d_raw.strip():
                            label_defs[lbl] = d_raw.strip()

                # --- Bouton: Générer automatiquement pour les labels saisis ---
                if emb_card.button(
                    "✨ Générer les définitions à partir des exemples saisis (Ollama)",
                    use_container_width=True,
                    key="btn_gen_defs_manual"
                ):
                    shots_present = {k: v for k, v in shots.items() if v}
                    if not shots_present:
                        emb_card.warning("Aucun exemple saisi. Ajoute quelques lignes avant de générer.")
                    else:
                        few_shot_embedding.set_shots(shots_present)

                        try:
                            defs_map = few_shot_embedding.set_definitions(
                                allow_defs=True,
                                label_defs=None,
                                model_defs="mistral:latest",
                                max_terms_defs=10,
                                temperature_defs=0.0,
                            )
                        except Exception as e:
                            defs_map = {k: "" for k in shots_present}
                            emb_card.warning(f"LLM indisponible, fallback local utilisé. Détails: {e}")

                        # FIX: buffer puis appliquer au prochain run
                        st.session_state["pending_def_map"] = defs_map
                        st.session_state["apply_def_map"] = True
                        st.rerun()



        # --------- Aperçu des prototypes ---------
        with emb_card.expander("Aperçu des prototypes (shots)"):
            if shots:
                preview_rows = [
                    {"label": k, "n_exemples": len(vs), "exemple_1": (vs[0] if vs else "")}
                    for k, vs in shots.items()
                ]
                emb_card.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
            else:
                emb_card.info("Aucun exemple disponible pour construire les prototypes.")

        # --------- Calibration des seuils ---------
        colA, colB = emb_card.columns([1, 1])
        with colA:
            calib_btn = st.button("🧪 Calibrer (leave-one-out)", use_container_width=True)
        with colB:
            run_embed_btn = st.button("🚀 Lancer (Embeddings)", type="primary", use_container_width=True)

        thr_default, mar_default = 0.35, 0.05
        thr_val, mar_val = thr_default, mar_default

        if calib_btn:
            try:

                 # >>> IMPORTANT : pousser les données dans l'expérience <<<
                few_shot_embedding.set_shots(shots)
                few_shot_embedding.set_definitions(allow_defs=False, label_defs=label_defs)



                few_shot_embedding.build_prototypes(alpha_def=None)
                thr_val, mar_val = few_shot_embedding.calibrate()
                emb_card.success(f"Seuil calibré ≈ {thr_val:.2f} | Marge ≈ {mar_val:.2f}")
            except Exception as e:
                emb_card.error(f"Calibration impossible : {e}")
        else:
            emb_card.caption(f"Seuil par défaut: {thr_val} | Marge par défaut: {mar_val}")

        # --------- Classification embeddings ---------
        if run_embed_btn:
            if not shots:
                emb_card.error("Aucun prototype. Fournis des exemples (CSV labellisé ou saisie manuelle).")
            else:
                try:

                    # >>> IMPORTANT : pousser les données dans l'expérience <<<
                    few_shot_embedding.set_shots(shots)
                    few_shot_embedding.set_definitions(allow_defs=False, label_defs=label_defs)



                    protos = few_shot_embedding.build_prototypes(alpha_def=None)

                    # Colonne à classer
                    if shot_source == "CSV labellisé":
                        classify_col = emb_text_col
                    else:
                        if default_text_col is not None and default_text_col in list(df.columns):
                            classify_col = default_text_col
                        else:
                            classify_col = emb_card.selectbox("Colonne texte à classer", options=list(df.columns))

                    data2 = df.copy()
                    if not do_full:
                        data2 = data2.head(sample_n).copy()

                    n_rows2 = len(data2)
                    prog2 = st.progress(0, text="Embeddings : classification en cours…")
                    res_labels, res_conf = [], []

                    for i, txt in enumerate(data2[classify_col].astype(str).fillna("").tolist()):
                        if use_multi:
                            r = emb_classify.classify_one_multi(txt, protos, per_label_threshold=max(0.4, thr_val),   embedder = few_shot_embedding.embedder, model_name=emb_model_name)
                            res_labels.append(", ".join(r["labels"]))
                            best = max((list(r["sims"].values()) or [0.0]))
                            res_conf.append(float(best))
                        else:
                            r = emb_classify.classify_one(txt, protos, threshold=thr_val, margin=mar_val, allow_other=allow_other, embedder = few_shot_embedding.embedder, model_name=emb_model_name)
                            res_labels.append(r["label"])
                            res_conf.append(r["confidence"])

                        prog2.progress((i + 1) / n_rows2, text=f"Embeddings… ({i+1}/{n_rows2})")

                    data2["emb_label"] = res_labels
                    data2["emb_confidence"] = res_conf

                    with ui_helper.card_block() as out_card:
                        out_card.success("Terminé (Embeddings) ✅")
                        out_card.dataframe(data2, use_container_width=True)

                        if not use_multi:
                            counts2 = data2["emb_label"].value_counts(dropna=False)
                            out_card.markdown("#### Répartition des labels (Embeddings)")
                            out_card.dataframe(pd.DataFrame({"label": counts2.index, "count": counts2.values}), use_container_width=True)

                        csv2 = data2.to_csv(index=False).encode("utf-8")
                        out_card.download_button(
                            "💾 Télécharger CSV (Embeddings)",
                            data=csv2,
                            file_name="classified_results_embeddings.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                except Exception as e:
                    emb_card.error(f"Erreur embeddings : {e}")
