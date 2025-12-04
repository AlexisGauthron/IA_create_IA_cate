from __future__ import annotations

CUSTOM_CSS = """
<style>
/* --- Thème de fond --- */
:root{
  --card-bg: rgba(20, 24, 38, .55);
  --card-border: rgba(255,255,255,0.06);
  --text-strong: #e5e7eb;
  --text-muted:  #a3a3a3;
}
.main { background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%); }
.block-container { padding-top: 2rem; }

/* --- Card générique (si tu utilises <div class="card">) --- */
.card{
  position: relative;          /* important: contexte pour les éléments en absolute */
  overflow: hidden;            /* important: clipper tout ce qui dépasse */
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;

  background: var(--card-bg);  /* (remplace l'ancien vert) */
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 20px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.20);
}
/* Sécu: empêcher images/iframes de “pousser” hors de la card */
.card * { max-width: 100%; }

/* Bande / glow décoratif utilisable dans la card */
.card .ribbon{
  position: absolute;
  left: 16px; right: 16px; top: 12px;
  height: 12px;
  border-radius: 999px;
  filter: blur(6px);
  pointer-events: none; /* pas cliquable */
}
/* Utilitaire: forcer pleine largeur interne */
.card .fullwidth{ width: 100% !important; }

/* --- Styler automatiquement les containers Streamlit en "card"
       (utile avec: with st.container(border=True): ... ) ------------------- */
div[data-testid="stContainer"] > div:has(> div[data-testid="stVerticalBlock"]) {
  position: relative;
  overflow: hidden;
  border-radius: 18px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  padding: 16px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.20);
}

/* Variante plus claire pour la sidebar (facultatif) */
section[data-testid="stSidebar"] div[data-testid="stContainer"] > div:has(> div[data-testid="stVerticalBlock"]) {
  background: rgba(255,255,255,0.06);
}

/* --- Typo & divers --- */
h1, h2, h3, h4, label, .stMarkdown { color: var(--text-strong); }
small, .help { color: var(--text-muted); }

.badge{
  display:inline-flex; align-items:center; gap:.5rem; padding:.25rem .6rem;
  border-radius:999px; background: rgba(255,255,255,0.07);
  color: var(--text-strong); font-size:.8rem;
}

hr{ border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0; }
a, .stDownloadButton { text-decoration: none; }
</style>
"""
