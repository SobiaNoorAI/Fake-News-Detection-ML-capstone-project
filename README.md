# Fake News Detection - Data Seekho Project

## 📌 Overview
This project is a **Fake News Detection** system using **machine learning models** to classify news articles as **real or fake**. The dataset used is sourced from **Hugging Face: ErfanMoosaviMonazzah/fake-news-detection-dataset-English**. The project applies **NLP techniques** for text preprocessing, feature extraction, and classification using **Naive Bayes, Logistic Regression, and Random Forest** models.


![Project](https://github.com/dataseekho/fake-news-detection/blob/main/images/Fake%20News%20Detection%20-%20Project%20Diagram.png)
---

## 📖 Table of Contents
1. [Installation](#%EF%B8%8F-installation)
2. [Dataset](#-dataset)
3. [Preprocessing](#%EF%B8%8F-preprocessing)
4. [Exploratory Data Analysis](#-exploratory-data-analysis)
5. [Feature Engineering](#-feature-engineering)
6. [Model Training](#-model-training)
7. [Evaluation](#-model-evaluation)
8. [Usage](#-usage)
9. [Contributing](#-contributing)
10. [License](#-license)

---

## ⚙️ Installation
To run this project locally, install the necessary dependencies:

```bash
pip install datasets pandas seaborn matplotlib nltk scikit-learn wordcloud
```

Ensure that necessary NLTK resources are downloaded:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 📊 Dataset
The dataset is loaded using the **Hugging Face datasets library**:

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")
df = pd.DataFrame(dataset['train'])
```

**Data Features:**
| Column | Description |
|--------|------------|
| `text`  | News article content |
| `label` | **0 (Fake News), 1 (Real News)** |

---

## 🛠️ Preprocessing
The dataset undergoes the following preprocessing steps:
- ✅ Removing special characters, URLs, and mentions
- ✅ Tokenization
- ✅ Stopword removal
- ✅ Lemmatization
- ✅ Lowercasing text

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

df["cleaned_text"] = df["text"].apply(clean_text)
```

---

## 🔍 Exploratory Data Analysis
EDA includes:
- Checking class distribution
- Word frequency analysis
- N-gram analysis
- Word cloud visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x="label", palette="viridis")
plt.title("Class Distribution")
plt.show()
```

---

## 🔬 Feature Engineering
TF-IDF vectorization is applied to convert text into numerical features:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]
```

---

## 🤖 Model Training
The dataset is split into **training and test sets**, and three classification models are trained:
✔️ **Naive Bayes**
✔️ **Logistic Regression**
✔️ **Random Forest**

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB().fit(X_train, y_train)
lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
```

---

## 📊 Model Evaluation
Models are evaluated based on **accuracy, classification report, and confusion matrix**:

```python
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_model(lr_model, X_test, y_test)
```

---

## 🚀 Usage
To predict fake news on new text:

```python
def predict_news(text, model):
    cleaned_text = clean_text(text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    return "Real News" if model.predict(vectorized_text)[0] == 1 else "Fake News"

sample_text = "This news is completely fake and misleading."
print("Prediction:", predict_news(sample_text, lr_model))
```

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve this project, follow these steps:
1. **Fork** the repository
2. **Create a new branch**
3. **Make your changes**
4. **Submit a pull request**

---

## 📜 License
This project is licensed under the **MIT License**. See the LICENSE file for details.

---

🚀 **Developed with ❤️ by the DataSeekho Team**

