import streamlit as st
import pandas as pd
from typing import List, Tuple

# Import preprocessing utilities
from nlpFunc.other_preprocessing import (
    remove_punctuation_with_whiteSpace,
    remove_digits,
    remove_stopWords,
    expand_user_query,
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

    with row1[3]:
        if st.button("🔎 Expand Query") and _require_text(text):
            result = expand_user_query(text)

            st.subheader("📌 Original Query")
            st.write(result['original_query'])

            st.subheader("📚 Synonym Expansions")
            st.text(result['synonym_expansions'])

            st.subheader("📝 Expanded Query")
            st.code(result['expanded_query'])

