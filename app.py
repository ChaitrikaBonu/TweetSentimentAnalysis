import streamlit as st
import pickle
import re
import pandas as pd
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
TEXT_COLUMN_CANDIDATES = (
    "text",
    "tweet",
    "tweet_text",
    "full_text",
    "content",
    "body",
)

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

def predict_sentiment(text):

    cleaned = clean_tweet(str(text))

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        return "Positive"

    return "Negative"

def find_text_column(columns):

    normalized_columns = {
        str(column).strip().lower(): column
        for column in columns
    }

    for column in TEXT_COLUMN_CANDIDATES:
        if column in normalized_columns:
            return normalized_columns[column]

    return None

# UI
st.title("Twitter Sentiment Analysis")

st.write("Enter a tweet and predict sentiment")

tweet = st.text_area("Tweet")

if st.button("Analyze Sentiment"):

    sentiment = predict_sentiment(tweet)

    if sentiment == "Positive":
        st.success("😊 Positive Sentiment")
    else:
        st.error("😞 Negative Sentiment")

st.markdown("---")
st.subheader("Analyze Tweets from CSV")
st.write("Upload a CSV with a tweet text column, such as text, tweet, tweet_text, full_text, content, or body.")

uploaded_file = st.file_uploader("Upload tweet CSV", type=["csv"])

if uploaded_file is not None:

    csv_data = pd.read_csv(uploaded_file)

    text_column = find_text_column(csv_data.columns)

    if text_column is None:
        st.error("CSV must include a text, tweet, tweet_text, full_text, content, or body column.")
    else:
        results = csv_data.copy()
        results["predicted_sentiment"] = (
            results[text_column]
            .fillna("")
            .astype(str)
            .map(predict_sentiment)
        )

        st.dataframe(
            results[[text_column, "predicted_sentiment"]].head(100),
            use_container_width=True,
        )
        st.download_button(
            "Download Predictions",
            results.to_csv(index=False).encode("utf-8"),
            "tweet_sentiment_predictions.csv",
            "text/csv",
        )
