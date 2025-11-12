from typing import Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
import pandas as pd
import streamlit as st

import src.front.ui_helper as ui_helper

def upload_csv_ui(parent_left) -> Optional[pd.DataFrame]:
    with ui_helper.card_block(parent_left) as left_card:
        left_card.subheader("1) Charger votre CSV")

        uploaded_files = left_card.file_uploader(
            "CSV à classer",
            type=["csv"],
            accept_multiple_files=True
        )

        if not uploaded_files:
            left_card.info("Chargez un ou plusieurs fichiers CSV pour commencer.")
            return None

        def _read_csv_any(f) -> pd.DataFrame:
            try: f.seek(0)
            except Exception: pass
            try: return pd.read_csv(f, sep=None, engine="python")
            except Exception: pass
            try:
                try: f.seek(0)
                except Exception: pass
                return pd.read_csv(f)
            except Exception: pass
            try:
                try: f.seek(0)
                except Exception: pass
                return pd.read_csv(f, sep=";")
            except Exception as e:
                raise e

        # Charger tous les fichiers
        named_dfs: List[Tuple[str, pd.DataFrame]] = []
        for uf in uploaded_files:
            try:
                df_i = _read_csv_any(uf)
                df_i.columns = [str(c).strip() for c in df_i.columns]
                named_dfs.append((Path(uf.name).stem, df_i))
            except Exception as e:
                left_card.error(f"Impossible de lire « {uf.name} » : {e}")
                return None

        # Un seul fichier -> affichage direct
        if len(named_dfs) == 1:
            name, df = named_dfs[0]
            left_card.caption(f"Fichier: {name} — Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
            left_card.dataframe(df, use_container_width=True)
            return df

        # Vérifier >= 1 colonne commune à TOUS les fichiers
        dfs = [d for _, d in named_dfs]
        common_cols = set(dfs[0].columns)
        for d in dfs[1:]:
            common_cols &= set(d.columns)

        if not common_cols:
            left_card.error(
                "Plusieurs fichiers chargés, mais **aucun nom de colonne commun** à tous les fichiers. "
                "Veuillez renommer au moins une colonne identiquement dans chaque CSV."
            )
            with left_card.expander("Colonnes détectées par fichier"):
                for n, d in named_dfs:
                    st.write(f"**{n}** → {list(d.columns)}")
            return None

        # 1) Concaténation (union des colonnes)
        merged = pd.concat(dfs, axis=0, join="outer", ignore_index=True)

        # 2) Supprimer les doublons exacts
        before_exact = len(merged)
        merged = merged.drop_duplicates(keep="first")
        removed_exact = before_exact - len(merged)

        # 3) Supprimer les doublons “partiels” (lignes dominées par une ligne plus complète)
        before_wild = len(merged)
        merged = _drop_wildcard_dominated_rows(merged)
        removed_wild = before_wild - len(merged)

        left_card.success(
            f"Concaténation réussie ({len(named_dfs)} fichiers). "
            f"Résultat: {merged.shape[0]} lignes × {merged.shape[1]} colonnes "
            f"(doublons exacts supprimés: {removed_exact}, "
            f"doublons partiels supprimés: {removed_wild})."
        )
        left_card.dataframe(merged, use_container_width=True)
        return merged


def _drop_wildcard_dominated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes redondantes 'partielles':
    - Une ligne A est supprimée s'il existe une ligne B telle que
      pour TOUTE colonne non vide de A, la valeur de B est identique,
      et B a >= autant de champs renseignés (souvent plus).
    - Les lignes totalement vides sont supprimées.
    - Préserve la 1ère occurrence des lignes les plus complètes.
    Implémentation:
      - normalise en chaînes,
      - trie par nb de champs non vides (desc),
      - index inversé (col,val) -> set(rows gardées),
      - si l'intersection des sets pour toutes les paires (col,val) de la ligne courante est non vide,
        alors la ligne est dominée -> on la supprime.
    """
    if df.empty:
        return df

    # Normaliser: trim, tout en str pour comparer proprement entre fichiers
    norm = df.copy()
    for c in norm.columns:
        norm[c] = norm[c].apply(lambda x: None if (pd.isna(x) or (isinstance(x, str) and x.strip() == "")) else str(x).strip())

    cols = list(norm.columns)

    # Construire liste (idx, non_null_count, pairs[(col, val), ...])
    rows_meta = []
    for i, row in norm.iterrows():
        pairs = [(c, row[c]) for c in cols if row[c] is not None]
        rows_meta.append((i, len(pairs), pairs))

    # Trier: d'abord les plus COMPLETS (desc), ainsi ils "écrasent" les squelettes
    rows_meta.sort(key=lambda t: t[1], reverse=True)

    keep_mask = pd.Series(False, index=norm.index)
    inverted = defaultdict(set)  # (col, val) -> {idx_kept}

    for idx, count, pairs in rows_meta:
        # Drop lignes totalement vides
        if count == 0:
            continue

        # Pour décider si dominée: intersection des sets par (col,val)
        candidate_sets = []
        ok = True
        for p in pairs:
            s = inverted.get(p)
            if not s:
                ok = False  # si un (col,val) n'existe chez aucun kept, il ne peut pas être dominé
                break
            candidate_sets.append(s)

        if ok and candidate_sets:
            inter = set.intersection(*candidate_sets)
        else:
            inter = set()

        if inter:
            # Il existe au moins une ligne gardée qui matche toutes les valeurs non vides -> dominée -> on skip
            continue

        # Sinon: on garde, et on enregistre ses paires
        keep_mask.loc[idx] = True
        for p in pairs:
            inverted[p].add(idx)

    # Retourne uniquement les lignes gardées
    kept = df.loc[keep_mask.index[keep_mask]].reset_index(drop=True)
    return kept
