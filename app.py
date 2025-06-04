# app.py
import streamlit as st
import nltk
from UIpages.tokenizer import show_tokenizer_ui
from UIpages.preprocessings import show_preprocessing_ui
from UIpages.transformer_segmenter import show_segmenter_ui
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('words')
    nltk.download('stopwords')  

# Call this function at the start of your app
download_nltk_data()
st.set_page_config(page_title="NLTK Playground", page_icon="🧠", layout="centered")

# Sidebar Navigation
st.sidebar.title("Navigation")

if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation buttons
if st.sidebar.button("🧪 Tokenizer"):
    st.session_state.page = "tokenizer"

if st.sidebar.button("🧹 Preprocessing"):
    st.session_state.page = "preprocessing"

if st.sidebar.button("✂️ TransformerSegmenter"):
    st.session_state.page = "segmenter"

# Routing
if st.session_state.page == "tokenizer":
    show_tokenizer_ui()

elif st.session_state.page == "preprocessing":
    show_preprocessing_ui()

elif st.session_state.page == "segmenter":
    show_segmenter_ui()

else:
    st.title("👋 Welcome")
    st.write("Please select a page from the sidebar.")
