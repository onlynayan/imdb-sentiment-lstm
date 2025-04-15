# 🎬 IMDB Sentiment Analysis using LSTM

Welcome to the **IMDB Sentiment Analysis** project! This deep learning-powered classifier can distinguish between **positive** and **negative** movie reviews using an LSTM neural network. It's trained on the popular Kaggle IMDB dataset.

---

## 📁 Dataset Overview

- **Source**: [Kaggle - IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 reviews
- **Balanced**: 25k positive, 25k negative

---

## 🧠 Model Architecture

The model uses the following layers:

- `Embedding`: Converts words into vector embeddings
- `LSTM`: Captures temporal patterns in text
- `Dense`: Final sigmoid layer for binary classification

```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
```

---

## 📊 Visualizations

### 🔹 Sentiment Distribution
*(Plot showing how many positive vs. negative reviews are in the dataset)*

### 🔹 Top Words in Reviews
*(Two side-by-side bar charts: most frequent words in positive and negative reviews)*

### 🔹 Model Performance Over Epochs
*(Accuracy and loss trends during training/validation)*

---

## 🚀 Getting Started

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the Classifier

```bash
python sentiment_analysis.py
```

---

## 📌 Predict Example

```python
new_review = "This movie was a masterpiece!"
sentiment = predict_sentiment(new_review)
print(f"The Review is: {sentiment}")
```

---

## 📈 Model Evaluation

- Accuracy: ~85% on test set
- Loss: Varies based on number of epochs and regularization

---

## 🧪 Libraries Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Seaborn / Matplotlib

---

## 💡 Future Improvements

- Use bidirectional LSTM for better context understanding
- Try GRU, BERT, or transformer-based models
- Build a web-based interface for real-time predictions

---

## 🌟 Contribute

Found a bug or have a suggestion? Feel free to open an issue or submit a pull request!

---

## 📜 License

MIT License