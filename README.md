# 🌏 Sentiment Analysis of Malaysia's Wikipedia Page

<img src="/images/malaysia_word_cloud.png" width="600" alt="Malaysia Word Cloud">  
*Visualization of most frequent terms in Malaysia's Wikipedia content*

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Findings](#-key-findings)
- [Methodology](#-methodology)
- [Results](#-results)
- [How to Use](#-how-to-use)
- [Dependencies](#-dependencies)
- [Files](#-files)

---

## 🌟 Project Overview

This project performs **sentiment analysis** on the English Wikipedia page about Malaysia ([en.wikipedia.org/wiki/Malaysia](https://en.wikipedia.org/wiki/Malaysia)). 
Using Natural Language Processing (NLP) and machine learning techniques, we:

1. Scraped and preprocessed 61,389 characters of Wikipedia text
2. Analyzed sentiment distribution across 470 sentences
3. Identified key topics through word frequency analysis
4. Evaluated two machine learning models (Logistic Regression and Naive Bayes) for sentiment classification

**Key Questions Answered:**
- How is Malaysia portrayed in Wikipedia's content?
- What are the most discussed topics about Malaysia?
- Can machine learning accurately classify sentiment in encyclopedic text?

---

## 🔍 Key Findings

### 1. Sentiment Distribution
<img src="/images/sentiment_distribution.png" width="400" alt="Sentiment Distribution">

- **Neutral Dominance (70.4%)**: Wikipedia maintains an objective tone
- **Positive vs Negative**: 21.5% positive vs 8.1% negative sentences
- **Example Positive**: "Malaysia has a newly industrialized market economy"
- **Example Negative**: "The country faces challenges with deforestation"

### 2. Top Topics Identified
<img src="/images/most_common_words.png" width="500" alt="Most Common Words">

| Category | Top Terms |
|----------|-----------|
| Governance | government, federal, state |
| Culture | Malay, Chinese, Malaysian |
| Geography | peninsula, Sarawak, east |

### 3. Model Performance
<div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 15px 0;">
  <img src="/images/logistic_regression.png" width="350" alt="Logistic Regression">
  <img src="/images/naive_bayes.png" width="350" alt="Naive Bayes">
</div>

| Metric | Logistic Regression | Naive Bayes |
|--------|---------------------|-------------|
| Accuracy | 71% | 71% |
| Positive F1-Score | 0.83 | 0.83 |
| Negative Recall | 0% | 0% |

**Challenge**: Both models failed to identify negative sentences due to class imbalance (only 38 negative vs 101 positive sentences).

---

## 🧠 Methodology

### Data Pipeline
1. **Collection**: Web-scraped Wikipedia using BeautifulSoup
2. **Preprocessing**:
   - Removed citations ([1], [2]) and special characters
   - Tokenized into 470 sentences and 5,188 meaningful words
3. **Analysis**:
   - Used TextBlob for sentiment scoring (-1 to +1)
   - Implemented TF-IDF vectorization for ML models

### Model Training
- Trained on 139 non-neutral sentences
- Evaluated using standard metrics (precision, recall, F1)
- **Limitation**: Small dataset (especially negative examples)

---

## 📊 Results

### Text Statistics
<img src="/images/text_statistics.jpg" width="450" alt="Text Statistics">

| Metric | Value |
|--------|-------|
| Original Characters | 61,389 |
| Processed Characters | 58,091 |
| Sentences | 470 |
| Meaningful Words | 5,188 |

### Live Prediction Example
<img src="/images/final_text_predict.jpg" width="450" alt="Prediction Demo">
*The model correctly classifies an example "malaysia is a good country" as Positive*

---

Developed by PRAGNYA REDDY , b.tech 3rd year student from branch 
Computer Science ( Data Science ) at NMIMS Hyderabad

