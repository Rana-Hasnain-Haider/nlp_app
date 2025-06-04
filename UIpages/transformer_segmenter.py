import streamlit as st
from nlpFunc.segmentation_transformer import segment  # Adjust import path as needed

def show_segmenter_ui():
    st.title("✂️ Sentence Segmenter")
    st.write("Break your text into meaningful segments based on semantic similarity.")

    # Text area with character limit
    max_chars = 1500
    text = st.text_area(
        "Enter your paragraph (max 1500 characters):",
        height=200,
        max_chars=max_chars,
        placeholder="Paste or type your text here..."
    )

    if st.button("🔍 Segment Text"):
        if not text.strip():
            st.warning("Please enter some text to segment.")
        else:
            with st.spinner("Segmenting..."):
                try:
                    segments = segment(text)
                    st.success(f"Segmented into {len(segments)} parts.")
                    for i, seg in enumerate(segments, 1):
                        st.code(seg, language="markdown")  # Easily copyable
                except Exception as e:
                    st.error(f"An error occurred during segmentation:\n{str(e)}")
