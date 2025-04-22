import nltk

# Ensure NLTK resources are downloaded BEFORE any Streamlit caching
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import requests
from bs4 import BeautifulSoup
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure required NLTK resources are downloaded
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Malaysia Sentiment Analysis", layout="wide")

def scrape_wikipedia(country):
    try:
        url = f"https://en.wikipedia.org/wiki/{country.replace(' ', '_')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        st.error(f"Error scraping Wikipedia: {str(e)}")
        return ""

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9. ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    if not text:
        return 'neutral'
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def prepare_data(df):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['sentence'])
    y = df['sentiment']
    
    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    return train_test_split(X_res, y_res, test_size=0.2, random_state=42), tfidf

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Naive Bayes': MultinomialNB(),
        'KNN': KNeighborsClassifier()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = round(acc * 100, 2)
    return results

def main():
    st.title("Malaysia Wikipedia Sentiment Analysis")

    # Ensure resources are available
    ensure_nltk_resources()

    @st.cache_data(show_spinner=False)
    def process_data():
        with st.spinner("Scraping and processing Malaysia Wikipedia page..."):
            raw_text = scrape_wikipedia("Malaysia")
            if not raw_text:
                st.error("Failed to fetch Wikipedia content")
                return None, None, None, None, None, None
            
            cleaned_text = clean_text(raw_text)
            sentences = sent_tokenize(cleaned_text)
            
            data = []
            for sentence in sentences:
                if len(sentence.split()) > 3:
                    sentiment = analyze_sentiment(sentence)
                    data.append({'sentence': sentence, 'sentiment': sentiment})
            
            df = pd.DataFrame(data)
            
            if df.empty:
                st.error("No valid sentences found after processing")
                return None, None, None, None, None, None
            
            words = word_tokenize(cleaned_text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
            filtered_text = ' '.join(filtered_words)
            
            try:
                (X_train, X_test, y_train, y_test), vectorizer = prepare_data(df.copy())
                models = train_models(X_train, y_train)
                model_results = evaluate_models(models, X_test, y_test)
                return df, filtered_text, model_results, models, vectorizer, raw_text
            except Exception as e:
                st.error(f"Error in machine learning processing: {str(e)}")
                return None, None, None, None, None, None
    
    df, filtered_text, model_results, models, vectorizer, raw_text = process_data()
    
    if df is None:
        return

    st.success("Data processing complete!")

    # Text stats
    st.subheader("Text Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Characters", len(raw_text))
    col2.metric("Total Sentences", len(sent_tokenize(clean_text(raw_text))))
    col3.metric("Processed Sentences", len(df))

    # WordCloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
    st.image(wordcloud.to_array(), use_column_width=True)

    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # Model results
    st.subheader("Model Accuracy Comparison")
    result_df = pd.DataFrame(model_results.items(), columns=["Model", "Accuracy"])
    st.dataframe(result_df.sort_values(by="Accuracy", ascending=False))

    # Show sample predictions
    st.subheader("Sample Sentences with Sentiment")
    st.dataframe(df.sample(10))

if __name__ == "__main__":
    main()
