import streamlit as st
import pandas as pd
from typing import List, Tuple

# Import preprocessing utilities
from nlpFunc.other_preprocessing2 import (
    Lancaster_stemmer,
    porter_stemmer,
    NER,
    lemmatize_sentence
)


def _require_text(text: str) -> bool:
    """Return *True* if *text* is non-empty and not just whitespace, else warn."""
    if text.strip():
        return True
    st.error("Please enter some text first.")
    return False


def _display_list(header: str, items: List[str]):
    st.subheader(header)
    st.write(items)


def _display_tuples(header: str, pairs: List[Tuple[str, str]]):
    st.subheader(header)
    if pairs:
        df = pd.DataFrame(pairs, columns=["Entity", "Label"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No named entities found.")

def show_preprocessing_ui2():
    """Streamlit page for common NLP preprocessing helpers."""

    st.title("🧹 Text Preprocessing Playground")
    text = st.text_area(
        "Enter your text:",
        height=150,
        value="Barack Obama was born in Hawaii on August 4, 1961.",
    )

    # ---------------------------------------------------------------------
    # 1st row of buttons
    # ---------------------------------------------------------------------
    row1 = st.columns(4)


    with row1[0]:
        if st.button("🌿 Lancaster Stemmer") and _require_text(text):
            stems = Lancaster_stemmer(text)
            _display_list("Lancaster Stems", stems)

    with row1[1]:
        if st.button("🌱 Porter Stemmer") and _require_text(text):
            stems = porter_stemmer(text)
            _display_list("Porter Stems", stems)

    with row1[2]:
        if st.button("🔤 WordNet Lemmatizer") and _require_text(text):
            lemas = lemmatize_sentence(text)
            _display_list("WordNet Lemas", lemas)

    with row1[3]:
        if st.button("🧑‍💼 Named Entities") and _require_text(text):
            entities = NER(text)
            _display_tuples("Named Entities", entities)

