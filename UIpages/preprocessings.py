import streamlit as st
import pandas as pd
from typing import List, Tuple

# Import preprocessing utilities
from nlpFunc.other_preprocessing import (
    remove_punctuation_with_whiteSpace,
    remove_digits,
    remove_stopWords,
    Lancaster_stemmer,
    porter_stemmer,
    pos_tagging,
    NER,
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


def show_preprocessing_ui():
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
        if st.button("✂️ Remove Punctuation") and _require_text(text):
            cleaned = remove_punctuation_with_whiteSpace(text)
            st.subheader("Without Punctuation")
            st.write(cleaned)

    with row1[1]:
        if st.button("🔢 Remove Digits") and _require_text(text):
            cleaned = remove_digits(text)
            st.subheader("Without Digits")
            st.write(cleaned)

    with row1[2]:
        if st.button("🚫 Stop-Words") and _require_text(text):
            # The helper expects an *iterable* of documents, not a single str.
            stop_free = remove_stopWords([text])
            _display_list("Stop-word-free Tokens", list(stop_free))

    # ---------------------------------------------------------------------
    # 2nd row of buttons
    # ---------------------------------------------------------------------
    row2 = st.columns(4)

    with row2[0]:
        if st.button("🌿 Lancaster Stemmer") and _require_text(text):
            stems = Lancaster_stemmer(text)
            _display_list("Lancaster Stems", stems)

    with row2[1]:
        if st.button("🌱 Porter Stemmer") and _require_text(text):
            stems = porter_stemmer(text)
            _display_list("Porter Stems", stems)

    with row2[2]:
        if st.button("🏷️ POS Tagging") and _require_text(text):
            tags = pos_tagging(text)
            st.subheader("POS Tags")
            st.write(tags)

    with row2[3]:
        if st.button("🧑‍💼 Named Entities") and _require_text(text):
            entities = NER(text)
            _display_tuples("Named Entities", entities)


