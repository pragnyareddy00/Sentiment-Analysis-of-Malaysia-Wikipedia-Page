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

# Download NLTK resources
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
    """Scrape text from Wikipedia page of the given country."""
    try:
        url = f"https://en.wikipedia.org/wiki/{country}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        
        return text
    except Exception as e:
        st.error(f"Error scraping Wikipedia: {e}")
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
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def create_wordcloud(text):
    """Generate word cloud from text."""
    if not text:
        return plt.figure()
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
    if df.empty:
        return None, None, None, None, None
    
    # Filter only positive and negative sentiments
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    
    if df.empty:
        return None, None, None, None, None
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['sentence'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to balance classes
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except Exception as e:
        st.warning(f"SMOTE failed: {e}. Using original data.")
        X_train_res, y_train_res = X_train, y_train
    
    return X_train_res, X_test, y_train_res, y_test, vectorizer

def train_models(X_train, y_train):
    """Train multiple machine learning models."""
    if X_train is None or y_train is None:
        return {}
    
    models = {
        'Logistic Regression': LogisticRegression(),
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
            st.warning(f"Failed to train {name}: {e}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return accuracy scores."""
    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        except Exception as e:
            st.warning(f"Failed to evaluate {name}: {e}")
            results[name] = 0.0
    return results

def main():
    st.title("Malaysia Wikipedia Sentiment Analysis")
    
    # Scrape and process data
    with st.spinner("Scraping and processing Malaysia Wikipedia page..."):
        raw_text = scrape_wikipedia("Malaysia")
        if not raw_text:
            st.error("Failed to scrape Wikipedia page. Please try again later.")
            return
            
        cleaned_text = clean_text(raw_text)
        sentences = sent_tokenize(cleaned_text) if cleaned_text else []
        
        # Create dataframe of sentences and sentiment
        data = []
        for sentence in sentences:
            if len(sentence.split()) > 3:  # Filter very short sentences
                sentiment = analyze_sentiment(sentence)
                data.append({'sentence': sentence, 'sentiment': sentiment})
        
        df = pd.DataFrame(data)
        
        if df.empty:
            st.error("No valid sentences found for analysis.")
            return
            
        # Word tokenization and stopwords removal
        words = word_tokenize(cleaned_text.lower()) if cleaned_text else []
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
        filtered_text = ' '.join(filtered_words)
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, vectorizer = prepare_data(df.copy())
        if X_train is None:
            st.error("Not enough data for machine learning analysis.")
            return
            
        models = train_models(X_train, y_train)
        if not models:
            st.error("No models were successfully trained.")
            return
            
        model_results = evaluate_models(models, X_test, y_test)
    
    # Display results
    st.success("Data processing complete!")
    
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
    frequent_words = get_frequent_words(cleaned_text)
    if frequent_words:
        freq_df = pd.DataFrame(frequent_words, columns=['Word', 'Frequency'])
        st.dataframe(freq_df)
    else:
        st.warning("No frequent words found.")
    
    # Model performance
    st.subheader("Model Performance")
    if model_results:
        model_df = pd.DataFrame.from_dict(model_results, orient='index', columns=['Accuracy'])
        st.dataframe(model_df.sort_values('Accuracy', ascending=False))
    else:
        st.warning("No model results to display.")
    
    # Sentiment prediction
    st.subheader("Predict Sentiment of New Text")
    if models and vectorizer:
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
                st.success(f"Predicted Sentiment: {prediction.capitalize()} ")
            else:
                st.error(f"Predicted Sentiment: {prediction.capitalize()} ")
    else:
        st.warning("Model prediction not available due to previous errors.")

if __name__ == "__main__":
    main()
