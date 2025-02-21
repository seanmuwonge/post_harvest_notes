# -*- coding: utf-8 -*-
import streamlit as st
from transformers import pipeline
import json
import requests

# Load the QA model
qa_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')


# Function to extract relevant context from the pre-loaded JSON data
def extract_context_from_json(json_data, keyword):
    context = []
    for entry in json_data:
        if 'heading' in entry and 'content' in entry and keyword.lower() in entry['heading'].lower():
            if isinstance(entry['content'], list):
                context.extend(entry['content'])
            elif isinstance(entry['content'], str):
                context.append(entry['content'])
    context_str = ' '.join(context)
    return context_str if context_str.strip() != "" else "No relevant context found."

# Streamlit UI setup
st.title("QA Model for Various Data")

# File URL from GitHub (replace this URL with your GitHub raw URL)
github_url = "https://github.com/seanmuwonge/post_harvest_notes/blob/main/PH_EN.json"

# Function to fetch JSON file from GitHub
def load_json_from_github(url):
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {}

# Load the JSON data from GitHub
json_data = load_json_from_github(github_url)

# Ensure JSON data is loaded correctly
if not json_data:
    st.error("Error loading JSON data from GitHub.")
else:
    # Input fields for user question and keyword
    question = st.text_input("Ask a question:", "")
    keyword = st.text_input("Enter a keyword to extract context (leave blank for all content):", "")

    if question:
        # Extract context from JSON based on keyword (or use all content if no keyword is provided)
        if keyword.strip():
            context = extract_context_from_json(json_data, keyword)
        else:
            # If no keyword, use all content in the JSON
            context = ' '.join([entry['content'] for entry in json_data if 'content' in entry])

        if context.strip() == "No relevant context found.":
            st.write("No relevant context found.")
        else:
            # Answer the question using the QA model
            answer = qa_model(question=question, context=context)
            st.write("Answer:", answer['answer'])
            st.write("Confidence Score:", answer['score'])

