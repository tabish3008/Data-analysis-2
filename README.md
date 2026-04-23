#  Assignment Title

## (1) Problem Statement
Analyze public opinion about Zepto using tweets.
The challenge is to automatically classify short, noisy text (tweets) into positive, negative, or neutral sentiment, which is difficult due to slang, mixed opinions, and limited context.

## (2) Objective
- Build a machine learning model to classify tweet sentiment
Compare performance of multiple classifiers
Evaluate models using precision, recall, and accuracy
Understand customer perception of Zepto services

## (3) Dataset
Source:
Tweets collected using snscrape (Twitter scraping tool)
Features:
Tweet → raw text data
Sentiment → manually labeled (positive / negative / neutral)
Size:
Total: 100 tweets
Training set: 80
Testing set: 20

## (4) Methodology
Data Preprocessing
Converted text to lowercase
Removed URLs, mentions, hashtags, and special characters
Cleaned and standardized text data
2. Exploratory Data Analysis (EDA)
Checked class distribution (positive / negative / neutral)
Visualized sentiment distribution using bar chart
3. Model Building
Applied machine learning models:

Naïve Bayes
Support Vector Machine (SVM)

Used TF-IDF vectorization to convert text into numerical features.

4. Evaluation

Models evaluated using:
Accuracy → Overall correctness of predictions
Precision → How many predicted positives are actually correct
Recall → How many actual positives are correctly identified
Confusion Matrix → Detailed breakdown of prediction performance

These metrics provide a balanced view of model performance, especially for multi-class sentiment classification.

## (5) Results
- Model Performance Summary
Naïve Bayes
Fast and simple model
Lower accuracy due to independence assumption
Struggles with complex language patterns

## (6) How to Run
```bash
pip install -r requirements.txt
python main.py
🔧 Alternative (Jupyter Notebook)
jupyter notebook

Open:

notebooks/sentiment_analysis.ipynb
```

## (7) Conclusion
This project demonstrates that machine learning models can effectively classify sentiment from textual data such as tweets.

## (8) Student's details
- Name: Tabish syed 
- Roll No: 72
- UIN:231A001
- YEAR: TE-AIDS
