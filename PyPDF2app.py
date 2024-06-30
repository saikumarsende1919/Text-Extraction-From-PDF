import streamlit as st
import PyPDF2
import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_director_info(pdf_path):
    text = extract_text_from_pdf(pdf_path)

    din_pattern = r"\b\d{8}\b"
    director_type_pattern = r"(independent|executive)"
    name_pattern = r"Mr\.?\s[A-Z][a-z]+\s[A-Z][a-z]+"

    director_names = set(re.findall(name_pattern, text))
    director_types = set(re.findall(director_type_pattern, text, re.IGNORECASE))
    DINs = set(re.findall(din_pattern, text))

    # Use spaCy for part-of-speech tagging
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Count occurrences of different parts of speech
    pos_counts = {}
    for token in doc:
        pos = token.pos_
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts[pos] = 1

    return director_names, director_types, DINs, pos_counts

def generate_word_cloud(data, fig, ax):
    text = " ".join(data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def main():
    st.title("Information Extractor(PyPDF2)")
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        director_names, director_types, DINs, pos_counts = extract_director_info("temp.pdf")

        max_len = max(len(director_names), len(director_types), len(DINs))

        # Pad shorter lists with empty strings
        director_names = list(director_names) + [''] * (max_len - len(director_names))
        director_types = list(director_types) + [''] * (max_len - len(director_types))
        DINs = list(DINs) + [''] * (max_len - len(DINs))

        data = {
            "Director Names": director_names,
            "Director Types": director_types,
            "DINs": DINs
        }

        df = pd.DataFrame(data)

        st.write(df)

        fig, ax = plt.subplots()
        st.subheader("Director Names Word Cloud:")
        generate_word_cloud(director_names, fig, ax)

        fig, ax = plt.subplots()
        st.subheader("Director Types Word Cloud:")
        generate_word_cloud(director_types, fig, ax)

        st.subheader("Part-of-Speech Counts:")
        st.write(pos_counts)

if __name__ == "__main__":
    main()
