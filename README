# Twitter Sentiment Analysis

This project analyzes the sentiment of tweets using a **Bidirectional LSTM neural network**. The model predicts whether a tweet is **positive** or **negative**.

---

## Project Overview

- **Goal:** Classify tweets by sentiment.
- **Model:** Bidirectional LSTM using Keras.
- **Input:** Preprocessed tweets converted to sequences.
- **Output:** Binary sentiment label (0 = negative, 1 = positive).

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   cd YOUR_REPOSITORY
````

2. **Install dependencies**

   ```bash
   pip install keras tensorflow pandas numpy scikit-learn matplotlib nltk
   ```

3. **Run the notebook**

   ```bash
   jupyter notebook TWITTER_SENTIMENT.ipynb
   ```

---

## Model Summary

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

* **Embedding:** Turns words into dense vectors.
* **Bidirectional LSTM:** Learns from past and future context.
* **Dropout:** Reduces overfitting.
* **Dense layer:** Outputs sentiment prediction.

---

## Requirements

* Python 3.x
* Keras & TensorFlow
* Jupyter Notebook
* pandas, numpy, scikit-learn, nltk, matplotlib

