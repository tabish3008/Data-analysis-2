# ==============================
# Sentiment Analysis - Zepto
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns


# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv('data/tweets.csv')

print("\nDataset Loaded Successfully")
print(df.head())
print("\nDataset Shape:", df.shape)


# ==============================
# 2. Data Preprocessing
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['Clean_Tweet'] = df['Tweet'].apply(clean_text)


# ==============================
# 3. EDA - Sentiment Distribution
# ==============================
os.makedirs("results", exist_ok=True)

df['Sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution - Zepto")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("results/sentiment_distribution.png")
plt.show()


# ==============================
# 4. Train-Test Split
# ==============================
X = df['Clean_Tweet']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Size:", len(X_train))
print("Test Size:", len(X_test))


# ==============================
# 5. TF-IDF Vectorization
# ==============================
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nFeature Size:", X_train_vec.shape)


# ==============================
# 6. Naïve Bayes Model
# ==============================
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


# =========

# ==============================
# 8. SVM Model
# ==============================
svm_model = SVC()
svm_model.fit(X_train_vec, y_train)

svm_pred = svm_model.predict(X_test_vec)

print("\n--- SVM ---")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


# ==============================
# 9. Confusion Matrix (SVM)
# ==============================
cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Neu','Pos'],
            yticklabels=['Neg','Neu','Pos'])

plt.title("Confusion Matrix - SVM")
plt.savefig("results/confusion_matrix.png")
plt.show()


# ==============================
# 10. Accuracy Comparison
# ==============================
models = ['Naive Bayes', 'Logistic Regression', 'SVM']
accuracies = [
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, svm_pred)
]

plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("results/accuracy_chart.png")
plt.show()


# ==============================
# 11. Sample Prediction
# ==============================
sample = ["Zepto delivery was very fast and amazing"]

sample_clean = [clean_text(sample[0])]
sample_vec = vectorizer.transform(sample_clean)

prediction = svm_model.predict(sample_vec)

print("\nSample Prediction:")
print("Tweet:", sample[0])
print("Predicted Sentiment:", prediction[0])
