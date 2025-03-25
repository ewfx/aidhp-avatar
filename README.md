# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
Welcome to our **Hyper-Personalized Financial Recommendation System** —an AI-powered solution designed to deliver tailored financial product suggestions based on customer profiles, transaction history, and social media sentiment.

Our system leverages machine learning and natural language processing (NLP) to:

✅ *Recommend* the most suitable financial products (loans, investments, credit cards)

✅ *Assess risk profiles* (High/Medium/Low) for better decision-making

✅ *Detect biases* in recommendations to ensure fairness

✅ *Explain* recommendations in natural language for transparency

### ✨ Key Features

- **Hybrid Recommendation Engine**
  - Combines Gradient Boosting with Transformer embeddings
  - Processes both structured and unstructured data
- **Risk Assessment**
  - Calculates personalized risk profiles (High/Medium/Low)
- **Bias Detection**
  - Monitors fairness across protected attributes
- **Explainable AI**
  - Provides natural language explanations for recommendations


## 🎥 Demo
 


## 💡 Inspiration

### Why We Built This

Financial institutions today face two critical challenges:

1. **Generic Recommendations** – Most systems suggest the same popular products to everyone, missing unique customer needs.
2. **Hidden Biases** – Algorithms may unintentionally discriminate based on gender, age, or location.

We were inspired by:
- Open Banking initiatives that empower customers with personalized financial insights.
- Ethical AI frameworks ensuring fairness in automated decisions.
- Behavioral Economics research showing people prefer explanations for recommendations.

### Real-World Impact

Our system addresses these gaps by:
- 🔹 **Personalizing suggestions** using transaction patterns and social context (e.g., a customer tweeting about home buying gets mortgage offers).
- 🔹 **Detecting Bias** in real-time (e.g., flagging if loans are disproportionately recommended to one demographic).
- 🔹 **Explaining recommendations** in plain language (e.g., "We suggest this loan because of your stable income and low existing debt").

### Use Cases

- 🏦 **Banks**: Increase conversion rates with hyper-relevant offers.
- 📱 **FinTech Apps**: Stand out with transparent, ethical AI.
- 🌍 **Emerging Markets**: Help underserved customers access tailored financial products.

### Example Scenario

**A small business owner in Nairobi:**

- **Traditional System**: Sends generic "business loan" ads.
  
- **Our System**: Recommends:
  - A low-collateral loan (based on their cash flow history).
  - Insurance bundles (because their social media mentions expanding to risky areas).
  - Explains: *"These match your 2-year revenue growth (+25%) and expansion plans."*

## ⚙️ What It Does
Our hyper-personalized recommendation system analyzes customer profiles, transaction histories, and social media sentiment to:

Suggest tailored financial products (loans, investments, credit cards)

Calculate individual risk profiles (High/Medium/Low)

Detect potential biases in recommendations

Provide natural language explanations for each suggestion

Key features:

Processes both structured (income, age) and unstructured data (social media posts)

Combines traditional ML with transformer-based embeddings

Adapts to severe class imbalances in the data

Delivers API endpoints for easy integration

## 🛠️ How We Built It
Core Architecture
1. Data Pipeline

 - Automated loading/cleaning of multiple data sources

 - Advanced feature engineering (numeric, categorical, text)

 - Sentiment analysis integration

2. Hybrid Recommendation Engine
```sh
graph TD
  A[Customer Data] --> B[Feature Engineering]
  B --> C[Gradient Boosting Classifier]
  A --> D[Text Embeddings]
  D --> C
  C --> E[Calibration]
  E --> F[Recommendations]
```
3. Fairness Monitoring

 - Continuous bias detection

 - Protected attribute analysis

 - Recommendation auditing

## 🚧 Challenges We Faced
### 1. Data Imbalance

1005 "general" vs 1-5 samples for specific products

Solution: Implemented SMOTE + BalancedRandomForest

### 2. Cold Start Problem

New customers with minimal history

Solution: Fallback to demographic-based rules

### 3. Feature Integration

Combining tabular data with text embeddings

Solution: Custom pipeline with dimension alignment

### 4. Real-time Performance

Transformer model latency

Solution: Caching + pre-computed embeddings

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/aidhp-avatar.git
   ```
2. Go to aidhp-avatar/code/src/ and open command prompt in this path and run below commands
```sh
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```
3. Install dependencies
```sh
pip install -r requirements.txt
```

4. Download NLTK data
```sh
python -m nltk.downloader vader_lexicon
```

5. Launch service
```sh
uvicorn app:app --reload
```

## 🏗️ Tech Stack
- 🔹 Framework: Uvicorn
- 🔹 Backend: FastAPI
- 🔹 Other: ML Models (Scikit-learn, Sentence-Transformers), NLP (NLTK, HuggingFace Transformers), Fairness	(AIF360, Fairlearn)

## 👥 Team
- **Ganesh Patnala** - [https://github.com/syampatnala007](#)
- **Tilak Kalyan**
- **Syam Kumar**
- **Anil Kumar**
