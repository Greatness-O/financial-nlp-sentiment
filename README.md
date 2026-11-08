# Financial Sentiment Classification with NLP
This project examines financial sentiment classification by building and evaluating traditional and deep learning natural language processing models using the Financial PhraseBank dataset. The goal is to classify financial news sentences into positive, negative or neutral sentiments.

## Overview

| Model | Type | Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression | Traditional ML | 82.49% | 0.7613 |
| Updated CNN | Deep Learning | 81.77% | 0.7535 |
| Tuned CNN | Deep Learning | 76.85% | 0.7010 |
| Multinomial Naive Bayes | Traditional ML | 75.54% | 0.6788 |
| Tuned Bi-LSTM | Deep Learning | 61.22%% | 0.4784 |
| Majority Class Baseline | — | 62.1% | — |

Almost all models significantly outperform the majority class baseline.

## Key Findings

- **CNN outperforms Bi-LSTM** on short financial sentences, local n-gram pattern detection matters more than global sequence modelling for this corpus.
- **Logistic Regression rivals deep models** when properly tuned with TF-IDF features and SMOTE oversampling, a reminder that complexity isn't always better.
- **Class imbalance is a major challenge**: the dataset is 62% neutral. SMOTE was applied at the data level for traditional models, `class_weight` was applied at the algorithm level for deep models.

## Dataset

**Financial PhraseBank** (`Sentences_75Agree.txt`) - 3,453 sentences annotated by ≥75% of financial experts.

| Class | Count | Proportion |
|---|---|---|
| Neutral | 2,146 | 62.1% |
| Positive | 887 | 25.7% |
| Negative | 420 | 12.2% |


## Repo Structure

```
financial-nlp-sentiment/
│
├── notebooks/
│   └── financial_sentiment_classification.ipynb  
│
├── data/
│   └── Sentences_75Agree.txt  # Dataset (download separately — see Dataset section)
│
├── outputs/
│   └── figures/               # Confusion matrices, training curves, word clouds
│
├── requirements.txt
└── README.md
```


## Methods

### Traditional Models

Both models use a shared TF-IDF vectorisation pipeline followed by **SMOTE oversampling** on the training split to address class imbalance.

- **Logistic Regression** — tuned via `GridSearchCV` over regularisation strength (`C`) and solver.
- **Multinomial Naive Bayes** — tuned via `GridSearchCV` over Laplace smoothing (`alpha`).

### Deep Learning Models

Both models use a Keras Embedding layer on padded sequences, with `class_weight` during training to handle imbalance. Architecture search is done using **KerasTuner (Hyperband)**.

- **CNN** — two Conv1D layers with MaxPooling, a Dense layer and Dropout.
- **Bi-LSTM** — bidirectional LSTM layer with Dropout, followed by a Dense classification head.

Both deep models use **EarlyStopping** (monitoring validation loss) to prevent overfitting.

### Evaluation Metrics

Each model is evaluated on a test split (20%) using:
- Overall **accuracy**
- **Macro-averaged F1-score** (primary metric, accounts for class imbalance)
- Per-class precision, recall and F1 (via `classification_report`)
- **Confusion matrix** visualisations

