pip install streamlit pandas numpy sentence-transformers scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    df["Question_Embedding"] = df["Question_Embedding"].apply(lambda x: np.array(eval(x)))  # Convert stored embeddings
    return df

df = load_data()

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # Replace with a suitable model

embedding_model = load_model()

st.title("Health Q&A: Heart, Lung, and Blood Conditions")
st.write("Ask a question related to heart, lung, or blood health.")

user_question = st.text_input("Enter your question here:")
if st.button("Get Answer"):
    if user_question:
        # Generate embedding for user input
        user_embedding = embedding_model.encode([user_question])

        # Compute cosine similarity
        similarities = cosine_similarity(user_embedding, np.vstack(df["Question_Embedding"].values))[0]
        best_match_idx = np.argmax(similarities)
        best_match_score = similarities[best_match_idx]

        # Set a similarity threshold
        threshold = 0.75  # Adjust based on experimentation
        if best_match_score >= threshold:
            st.success(f"**Answer:** {df.iloc[best_match_idx]['Answer']}")
            st.write(f"_(Similarity Score: {best_match_score:.2f})_")
        else:
            st.warning("I apologize, but I don't have information on that topic yet. Could you please ask another question?")
    else:
        st.error("Please enter a question.")

# Optional: "Clear" button
if st.button("Clear"):
    st.experimental_rerun()
with st.expander("Commonly Asked Questions"):
    for question in df["Question"].sample(5):  # Show random FAQs
        st.write(f"• {question}")
rating = st.radio("Was this answer helpful?", ["Yes", "No"])
