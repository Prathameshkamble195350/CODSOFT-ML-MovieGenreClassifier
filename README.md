# CODSOFT-ML-MovieGenreClassifier
Movie Genre Classification using Machine Learning and Natural Language Processing (NLP). This project predicts movie genres from plot summaries using text preprocessing, TF-IDF vectorization, and classification algorithms. Developed as part of the CODSOFT Machine Learning Internship Program.
# 🎬 Movie Genre Classification using Machine Learning | CODSOFT

## 📌 Project Overview

This project is a Machine Learning-based Movie Genre Classification system developed as part of the CODSOFT Machine Learning Internship. The model predicts the genre of a movie based on its plot description using Natural Language Processing (NLP) techniques and TF-IDF feature extraction.

---

## 🎯 Objective

The main objective of this project is to automatically classify movie genres from textual descriptions using Machine Learning algorithms.

---

## 📂 Dataset

Dataset Files:

- train_data.txt
- test_data.txt

Dataset Features:

| Column | Description |
|----------|-------------|
| ID | Unique Movie ID |
| TITLE | Movie Title |
| GENRE | Movie Genre (Target Variable) |
| DESCRIPTION | Movie Plot Description |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- WordCloud
- Jupyter Notebook

---

## 🔄 Project Workflow

### 1️⃣ Data Collection

- Load training and testing datasets
- Verify dataset structure

### 2️⃣ Data Exploration

- Dataset Information
- Shape Analysis
- Missing Value Analysis
- Statistical Summary

### 3️⃣ Exploratory Data Analysis (EDA)

Performed various visualizations:

- Genre Distribution Analysis
- Genre Frequency Count
- Description Length Distribution
- Description Length by Genre
- Word Cloud Visualization
- Genre Percentage Distribution

### 4️⃣ Data Preprocessing

- Text Cleaning
- Handling Missing Values
- Preparing Text Data for Modeling

### 5️⃣ Feature Engineering

TF-IDF Vectorization was applied to convert text data into numerical vectors.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)
```

### 6️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 7️⃣ Model Training

Algorithm Used:

- Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### 8️⃣ Model Evaluation

Evaluation Metrics:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report

```python
from sklearn.metrics import classification_report

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
```

### 9️⃣ Model Saving

```python
import pickle

pickle.dump(model, open('genre_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
```

### 🔟 Genre Prediction

Predict movie genres for unseen movie descriptions.

---

## 📊 Visualizations Included

✔ Genre Distribution Bar Chart

✔ Genre Distribution Pie Chart

✔ Description Length Histogram

✔ Description Length Box Plot

✔ Word Cloud Visualization

✔ Average Description Length Analysis

✔ Genre Frequency Analysis

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/yourusername/Movie-Genre-Classification-CODSOFT.git
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn wordcloud
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Movie genre classification.ipynb
```

---

## 📈 Results

The trained model successfully predicts movie genres based on textual descriptions.

Performance Metrics:

- Accuracy Score
- Precision
- Recall
- F1 Score

---

## 🔮 Future Improvements

- Naive Bayes Classifier
- Random Forest Classifier
- XGBoost Classifier
- Hyperparameter Tuning
- Deep Learning Models (LSTM)
- Transformer Models (BERT)
- Flask Web Application Deployment
- Streamlit Deployment

---

## 💡 Skills Demonstrated

- Data Cleaning
- Data Analysis
- Data Visualization
- Natural Language Processing (NLP)
- Feature Engineering
- TF-IDF Vectorization
- Machine Learning
- Logistic Regression
- Model Evaluation
- Model Deployment

---

## 👨‍💻 Author

**Prathamesh Kamble**

B.Sc. Physics Graduate

Machine Learning Enthusiast

CODSOFT Machine Learning Intern

---

## ⭐ Internship Task

Task: Movie Genre Classification

Internship: CODSOFT Machine Learning Internship

Project Type: Natural Language Processing (NLP)

Status: Completed ✅

---

### If you found this project useful, please give it a ⭐ on GitHub!
