import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

# Load model and vectorizer
with open("news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def clean_input(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter a news article:")

if st.button("Predict"):
    cleaned = clean_input(user_input)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    st.write("### 🔍 This article is:", "🟢 Real" if prediction[0] == 1 else "🔴 Fake")
