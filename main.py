# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.

import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

# Input URLs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

# Placeholder for displaying messages
placeholder = st.empty()

# Button to process URLs
process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != '']
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid URL")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

# Input query for question-answering
query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:
        placeholder.text("You must process URLs first")

