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
A brief overview of your project and its purpose. Mention which problem statement are your attempting to solve. Keep it concise and engaging.

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
What inspired you to create this project? Describe the problem you're solving.

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
