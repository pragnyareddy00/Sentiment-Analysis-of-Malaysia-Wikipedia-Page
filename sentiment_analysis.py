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
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import requests
from bs4 import BeautifulSoup
import re
import nltk
import os

# Configure NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.data.path.append(nltk_data_path)

# Set page config
st.set_page_config(page_title="Malaysia Sentiment Analysis", layout="wide")

def scrape_wikipedia(country):
    """Scrape text from Wikipedia page of the given country."""
    try:
        url = f"https://en.wikipedia.org/wiki/{country}"
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
        return None

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
    if analysis.sentiment.polarity > 0.1:
        return 'positive'
    elif analysis.sentiment.polarity < -0.1:
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
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    X = vectorizer.fit_transform(df['sentence'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to balance classes
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        return X_train_res, X_test, y_train_res, y_test, vectorizer
    except:
        return X_train, X_test, y_train, y_test, vectorizer

def train_models(X_train, y_train):
    """Train multiple machine learning models."""
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
            st.warning(f"Failed to train {name}: {str(e)}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return accuracy scores."""
    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
        except:
            results[name] = {'accuracy': 0, 'report': None}
    return results

def main():
    st.title("Malaysia Wikipedia Sentiment Analysis")
    
    # Scrape and process data
    with st.spinner("Scraping and processing Malaysia Wikipedia page..."):
        raw_text = scrape_wikipedia("Malaysia")
        if raw_text is None:
            st.error("Failed to fetch data from Wikipedia. Please try again later.")
            return
            
        cleaned_text = clean_text(raw_text)
        try:
            sentences = sent_tokenize(cleaned_text)
        except Exception as e:
            st.error(f"Error in tokenizing sentences: {str(e)}")
            return
        
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
        words = word_tokenize(cleaned_text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
        filtered_text = ' '.join(filtered_words)
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, vectorizer = prepare_data(df.copy())
        if X_train is None:
            st.warning("Insufficient data for machine learning analysis.")
            ml_success = False
        else:
            models = train_models(X_train, y_train)
            model_results = evaluate_models(models, X_test, y_test)
            ml_success = True
    
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
    if ml_success:
        st.subheader("Model Performance")
        accuracy_scores = {name: res['accuracy'] for name, res in model_results.items()}
        model_df = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy'])
        st.dataframe(model_df.sort_values('Accuracy', ascending=False))
        
        # Sentiment prediction
        st.subheader("Predict Sentiment of New Text")
        selected_model = st.selectbox("Select Model", list(models.keys()))
        user_input = st.text_area("Enter a sentence to analyze:")
        
        if user_input and st.button("Predict"):
            # Preprocess input
            cleaned_input = clean_text(user_input)
            # Vectorize
            input_vec = vectorizer.transform([cleaned_input])
            # Predict
            model = models[selected_model]
            try:
                prediction = model.predict(input_vec)[0]
                if prediction == 'positive':
                    st.success(f"Predicted Sentiment: {prediction.capitalize()}")
                else:
                    st.error(f"Predicted Sentiment: {prediction.capitalize()}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
