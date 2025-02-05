import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load the dataset and embeddings
@st.cache_data
def load_data():
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=', '))
    return df

df = load_data()

# Load the embedding model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

model = load_model()

# Streamlit UI
st.title("Health FAQ Assistant")

user_question = st.text_input("Ask a question about heart, lung, or blood health:")

if st.button("Get Answer"):
    if user_question:
        # Generate embedding for the user's question
        user_question_embedding = model.encode(user_question, convert_to_tensor=True)

        # Calculate cosine similarity with all questions in the dataset
        similarities = util.pytorch_cos_sim(user_question_embedding, torch.tensor(np.stack(df['Question_Embedding'].values))).tolist()[0]
        
        # Find the most similar question and its index
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]

        # Set a similarity threshold (experiment with this value)
        threshold = 0.7

        # Display the answer if similarity is above the threshold
        if similarity_score >= threshold:
            st.write(f"**Answer:** {df.loc[most_similar_idx, 'Answer']}")
        else:
            st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
    else:
        st.write("Please enter a question.")
