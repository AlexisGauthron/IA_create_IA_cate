# -------------------------------------------------
# Standard libs & typing
# -------------------------------------------------
from contextlib import contextmanager

# -------------------------------------------------
# Third‑party libs
# -------------------------------------------------
import streamlit as st


# -------------------------------------------------
# UI Helpers
# -------------------------------------------------
@contextmanager
def card_block(parent=st, border=True):
    """A minimal wrapper to keep Streamlit widgets grouped like a card."""
    c = parent.container(border=border)
    with c:
        yield c


def _compute_select_index(columns: list[str], default_text_col: str | None) -> int:
    """Return the index for a selectbox given a list of column names and an optional default.
    Falls back to 0 if default is None or not present.
    """
    try:
        cols = list(columns)
        if default_text_col is not None and default_text_col in cols:
            return cols.index(default_text_col)
    except Exception:
        pass
    return 0
