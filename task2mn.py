import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import ast

openai.api_key =  st.secrets["mykey"]

# Load the embedding model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

model = load_model()

# Function to get embedding for a given text
def get_embedding(text):
    return model.encode(text, convert_to_tensor=True).cpu().numpy()

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return util.cos_sim(embedding1, embedding2).item()

# Load the dataset and embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)
    return df

df = load_data()

def find_best_answer(user_question):
    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    # Calculate cosine similarities for all questions in the dataset
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

    # Find the most similar question and get its corresponding answer
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    # Set a similarity threshold to determine if a question is relevant enough
    similarity_threshold = 0.6  # You can adjust this value

    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"

# Streamlit UI
st.title("Health FAQ Assistant")

user_question = st.text_input("Ask a question about heart, lung, or blood health:")

if st.button("Get Answer"):
    if user_question:
        answer = find_best_answer(user_question)
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please enter a question.")
