import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# UI
st.title("Twitter Sentiment Analysis")

st.write("Enter a tweet and predict sentiment")

tweet = st.text_area("Tweet")

if st.button("Analyze Sentiment"):

    cleaned = clean_tweet(tweet)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.success("😊 Positive Sentiment")
    else:
        st.error("😞 Negative Sentiment")
        