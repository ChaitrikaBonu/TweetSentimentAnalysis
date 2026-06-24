# Twitter Sentiment Analysis using NLP

## Overview

This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) and Machine Learning techniques. The model classifies tweets into Positive and Negative sentiments based on their textual content.

The project uses the Sentiment140 dataset and implements a complete machine learning pipeline including data preprocessing, feature extraction, model training, evaluation, and deployment through a Streamlit web application.

---

## Dataset

Dataset: Sentiment140

* Total Tweets: 1.6 Million
* Sentiment Labels:

  * 0 = Negative
  * 4 = Positive

For faster experimentation, a random sample of 50,000 tweets was used.

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit

---

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Stopword Removal
4. Stemming using Porter Stemmer
5. TF-IDF Feature Extraction
6. Train-Test Split
7. Logistic Regression Model Training
8. Model Evaluation
9. Streamlit Deployment

---

## Project Structure

Twitter-Sentiment-Analysis/

├── training.1600000.processed.noemoticon.csv

├── sentiment140_sample_50000.csv

├── sentiment140_cleaned_50000.csv

├── sentiment_model.pkl

├── vectorizer.pkl

├── app.py

├── sentiment_analysis.ipynb

├── requirements.txt

└── README.md

---

## Installation

Install required packages:

pip install pandas numpy nltk scikit-learn matplotlib seaborn streamlit

---

## Running the Project

Train the model and save:

* sentiment_model.pkl
* vectorizer.pkl

Run the Streamlit application:

streamlit run app.py

Open:

http://localhost:8501

---

## Model Performance

Algorithm Used:

* Logistic Regression

Feature Extraction:

* TF-IDF Vectorization

Evaluation Metrics:

* Accuracy Score
* Classification Report
* Confusion Matrix

Expected Accuracy:

* Approximately 75% to 80%

---

## Features

* Automatic tweet preprocessing
* Sentiment prediction in real time
* User-friendly Streamlit interface
* Machine Learning based classification
* Easy deployment and scalability

---

## Future Enhancements

* Add Neutral sentiment classification
* Deep Learning models using LSTM and BERT
* Live Twitter API integration
* Sentiment trend visualization dashboard
* Multi-language sentiment analysis

---

## Author

Developed as a Natural Language Processing and Machine Learning project for sentiment classification on Twitter data.
