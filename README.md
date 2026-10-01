# SMS Spam Detection

Classifies SMS messages as spam or ham (legitimate) using TF-IDF features and Logistic Regression.

## Features

- Trained on the SMS Spam Collection dataset (5,572 messages)
- Text preprocessing: lowercasing, stopword removal, Porter stemming
- TF-IDF vectorization (4,900-word vocabulary)
- **94.5% test accuracy** with Logistic Regression

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/-NLTK-154F3C?style=flat)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Project Structure

```
SMSspamcollectionproject.ipynb   # Full pipeline: load data, preprocess, vectorize, train, evaluate
spam.csv                         # SMS Spam Collection dataset
```

## Pipeline

1. Load and clean the dataset (`v1`/`v2` columns → `label`/`message`)
2. Map labels: `ham` → 0, `spam` → 1
3. Preprocess text: lowercase → tokenize → stem → remove stopwords/non-alphabetic tokens
4. Vectorize with `TfidfVectorizer`
5. Train/test split (80/20) and fit a Logistic Regression classifier
6. Evaluate with accuracy, confusion matrix, and classification report

## Getting Started

Open `SMSspamcollectionproject.ipynb` in Jupyter and run all cells. Requires `pandas`, `nltk`, and `scikit-learn`.
