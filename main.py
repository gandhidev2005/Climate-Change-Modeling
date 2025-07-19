# ------------------------------
# 1. Importing required libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from textblob import TextBlob

# ------------------------------
# 2. Load dataset
# ------------------------------
data = pd.read_csv('climate_nasa.csv')

# ------------------------------
# 3. Data Exploration
# ------------------------------
print('\nFirst five rows:')
print(data.head())

print('\nData Info:')
print(data.info())

print('\nStatistical Description:')
print(data.describe())

# ------------------------------
# 4. Handling Missing Values
# ------------------------------
print('\nMissing values per column:')
print(data.isnull().sum())

# Filling missing commentsCount with 0 assuming no replies
data['commentsCount'] = data['commentsCount'].fillna(0)

# Dropping rows with missing text as it's crucial for NLP
data = data.dropna(subset=['text'])

# ------------------------------
# 5. Sentiment Analysis Feature Engineering
# ------------------------------
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

data['sentiment'] = data['text'].apply(get_sentiment)

# Classifying sentiment as positive, negative, neutral
def classify_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['sentiment_label'] = data['sentiment'].apply(classify_sentiment)

# ------------------------------
# 6. Visualising Sentiment Distribution
# ------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment_label', data=data)
plt.title('Sentiment Label Distribution')
plt.savefig('sentiment_distribution.png')
plt.close()

# ------------------------------
# 7. Feature Selection
# ------------------------------
# Selecting numeric features + engineered sentiment for regression modeling
features = ['likesCount', 'commentsCount', 'sentiment']
X = data[features]
y = data['likesCount']  # Example: predicting likesCount as engagement metric

# ------------------------------
# 8. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 9. Modeling - Random Forest Regressor
# ------------------------------
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# ------------------------------
# 10. Evaluation
# ------------------------------
print('\nRandom Forest Regressor Performance:')
print('MAE:', mean_absolute_error(y_test, y_pred_rf))
print('MSE:', mean_squared_error(y_test, y_pred_rf))
print('R2 Score:', r2_score(y_test, y_pred_rf))

# ------------------------------
# 11. Modeling - Linear Regression
# ------------------------------
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print('\nLinear Regression Performance:')
print('MAE:', mean_absolute_error(y_test, y_pred_lr))
print('MSE:', mean_squared_error(y_test, y_pred_lr))
print('R2 Score:', r2_score(y_test, y_pred_lr))

# ------------------------------
# 12. Future Projection Sample
# ------------------------------
# For demonstration: predicting likes for a new comment sample
sample_input = pd.DataFrame({
    'likesCount': [0],
    'commentsCount': [5],
    'sentiment': [0.3]
})
future_pred = model_rf.predict(sample_input)
print('\nPredicted likes for sample input:', future_pred)

# ------------------------------
# 13. Save Processed Data
# ------------------------------
data.to_csv('processed_climate_nasa.csv', index=False)

print('\nClimate Change Modeling Internship Task Completed Successfully.')
