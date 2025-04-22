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

# Ensure NLTK data is downloaded properly
try:
    # Test if punkt and stopwords are available
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    # If not available, download them
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Malaysia Sentiment Analysis", layout="wide")

def scrape_wikipedia(country):
    """Scrape text from Wikipedia page of the given country."""
    url = f"https://en.wikipedia.org/wiki/{country}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all paragraphs
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    
    return text

def clean_text(text):
    """Clean and preprocess the text."""
    # Remove square brackets and citations
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9. ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob."""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def create_wordcloud(text):
    """Generate word cloud from text."""
    if not text:
        return plt.figure()  # Return empty figure if text is empty
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def get_frequent_words(text, n=20):
    """Get most frequent words from text."""
    if not text:
        return []
        
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    
    freq_dist = nltk.FreqDist(filtered_words)
    return freq_dist.most_common(n)

def prepare_data(df):
    """Prepare data for machine learning."""
    # Filter only positive and negative sentiments
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    
    # Check if we have enough data
    if len(df) < 10 or len(df['sentiment'].unique()) < 2:
        st.warning("Not enough data for model training. Need both positive and negative examples.")
        return None, None, None, None, None
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['sentence'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to balance classes if there are enough samples
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        return X_train_res, X_test, y_train_res, y_test, vectorizer
    except ValueError as e:
        st.warning(f"SMOTE error: {e}. Using original imbalanced data.")
        return X_train, X_test, y_train, y_test, vectorizer

def train_models(X_train, y_train):
    """Train multiple machine learning models."""
    if X_train is None or y_train is None:
        return {}
        
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Naive Bayes': MultinomialNB(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            st.warning(f"Error training {name}: {e}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return accuracy scores."""
    if not models or X_test is None or y_test is None:
        return {}
        
    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        except Exception as e:
            st.warning(f"Error evaluating {name}: {e}")
    return results

def process_data():
    """Process data and return necessary components for the app."""
    # Scrape and process data
    raw_text = scrape_wikipedia("Malaysia")
    cleaned_text = clean_text(raw_text)
    
    # Handle potential errors with tokenization
    try:
        sentences = sent_tokenize(cleaned_text)
    except LookupError:
        # Fallback to simple splitting if NLTK fails
        sentences = [s.strip() + '.' for s in cleaned_text.split('.') if s.strip()]
        st.warning("NLTK sentence tokenization failed. Using basic sentence splitting.")
    
    # Create dataframe of sentences and sentiment
    data = []
    for sentence in sentences:
        if len(sentence.split()) > 3:  # Filter very short sentences
            sentiment = analyze_sentiment(sentence)
            data.append({'sentence': sentence, 'sentiment': sentiment})
    
    df = pd.DataFrame(data)
    
    # Word tokenization and stopwords removal
    try:
        words = word_tokenize(cleaned_text.lower())
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # Fallback if NLTK fails
        words = cleaned_text.lower().split()
        stop_words = set(['the', 'and', 'is', 'of', 'to', 'in', 'a', 'for', 'with', 'as', 'by', 'on', 'at', 'that', 'this'])
        st.warning("NLTK word tokenization or stopwords failed. Using basic tokenization.")
    
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    filtered_text = ' '.join(filtered_words)
    
    # Prepare data for ML
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df.copy())
    models = train_models(X_train, y_train)
    model_results = evaluate_models(models, X_test, y_test)
    
    return df, filtered_text, model_results, models, vectorizer, raw_text, sentences

def main():
    st.title("Malaysia Wikipedia Sentiment Analysis")
    
    with st.spinner("Scraping and processing Malaysia Wikipedia page..."):
        try:
            df, filtered_text, model_results, models, vectorizer, raw_text, sentences = process_data()
            st.success("Data processing complete!")
        except Exception as e:
            st.error(f"An error occurred during data processing: {e}")
            st.stop()
    
    # Display raw text stats
    st.subheader("Text Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Characters", len(raw_text))
    col2.metric("Total Sentences", len(sentences))
    col3.metric("Processed Sentences", len(df))
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    
    # Word cloud
    st.subheader("Word Cloud")
    wordcloud_fig = create_wordcloud(filtered_text)
    st.pyplot(wordcloud_fig)
    
    # Frequent words
    st.subheader("Most Frequent Words")
    frequent_words = get_frequent_words(filtered_text)
    freq_df = pd.DataFrame(frequent_words, columns=['Word', 'Frequency'])
    st.dataframe(freq_df)
    
    # Model performance
    if model_results:
        st.subheader("Model Performance")
        model_df = pd.DataFrame.from_dict(model_results, orient='index', columns=['Accuracy'])
        st.dataframe(model_df.sort_values('Accuracy', ascending=False))
        
        # Sentiment prediction
        st.subheader("Predict Sentiment of New Text")
        if models:
            selected_model = st.selectbox("Select Model", list(models.keys()))
            user_input = st.text_area("Enter a sentence to analyze:")
            
            if user_input and st.button("Predict"):
                # Preprocess input
                cleaned_input = clean_text(user_input)
                # Vectorize
                input_vec = vectorizer.transform([cleaned_input])
                # Predict
                model = models[selected_model]
                prediction = model.predict(input_vec)[0]
                
                # Display result
                if prediction == 'positive':
                    st.success(f"Predicted Sentiment: {prediction.capitalize()}")
                else:
                    st.error(f"Predicted Sentiment: {prediction.capitalize()}")
        else:
            st.warning("No models available for prediction.")
    else:
        st.warning("Insufficient data for model training. Sentiment prediction is unavailable.")

if __name__ == "__main__":
    main()
