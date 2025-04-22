import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
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

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Malaysia Sentiment Analysis", layout="wide")

def scrape_wikipedia(country):
    """Scrape text from Wikipedia page of the given country."""
    try:
        url = f"https://en.wikipedia.org/wiki/{country.replace(' ', '_')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        
        return text
    except Exception as e:
        st.error(f"Error scraping Wikipedia: {str(e)}")
        return ""

def clean_text(text):
    """Clean and preprocess the text."""
    if not text:
        return ""
    # Remove square brackets and citations
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9. ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob."""
    if not text:
        return 'neutral'
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:  # Increased threshold for positive
        return 'positive'
    elif analysis.sentiment.polarity < -0.1:  # Increased threshold for negative
        return 'negative'
    else:
        return 'neutral'

# ... [rest of your functions remain the same until main()]

def main():
    st.title("Malaysia Wikipedia Sentiment Analysis")
    
    # Add caching to avoid reprocessing on every rerun
    @st.cache_data(show_spinner=False)
    def process_data():
        with st.spinner("Scraping and processing Malaysia Wikipedia page..."):
            raw_text = scrape_wikipedia("Malaysia")
            if not raw_text:
                st.error("Failed to fetch Wikipedia content")
                return None, None, None, None, None, None
            
            cleaned_text = clean_text(raw_text)
            sentences = sent_tokenize(cleaned_text)
            
            # Create dataframe of sentences and sentiment
            data = []
            for sentence in sentences:
                if len(sentence.split()) > 3:  # Filter very short sentences
                    sentiment = analyze_sentiment(sentence)
                    data.append({'sentence': sentence, 'sentiment': sentiment})
            
            df = pd.DataFrame(data)
            
            if df.empty:
                st.error("No valid sentences found after processing")
                return None, None, None, None, None, None
            
            # Word tokenization and stopwords removal
            words = word_tokenize(cleaned_text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
            filtered_text = ' '.join(filtered_words)
            
            try:
                # Prepare data for ML
                X_train, X_test, y_train, y_test, vectorizer = prepare_data(df.copy())
                models = train_models(X_train, y_train)
                model_results = evaluate_models(models, X_test, y_test)
                return df, filtered_text, model_results, models, vectorizer, raw_text
            except Exception as e:
                st.error(f"Error in machine learning processing: {str(e)}")
                return None, None, None, None, None, None
    
    df, filtered_text, model_results, models, vectorizer, raw_text = process_data()
    
    if df is None:
        return
    
    # Display results
    st.success("Data processing complete!")
    
    # Display raw text stats
    st.subheader("Text Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Characters", len(raw_text))
    col2.metric("Total Sentences", len(sent_tokenize(clean_text(raw_text))))
    col3.metric("Processed Sentences", len(df))
    
    # ... [rest of your display code remains the same]

if __name__ == "__main__":
    main()
