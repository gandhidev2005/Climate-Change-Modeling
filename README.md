# 🌍 Climate Change Modeling
This repository contains the **Climate Change Modeling** using Machine Learning and Natural Language Processing (NLP) techniques on NASA Climate Change Facebook comments dataset.

---
## 📁 Project Overview
The goal of this project is to:
* Understand public sentiment regarding climate change using Facebook comments.
* Preprocess and analyze the dataset for missing values and NLP features.
* Build regression models to predict engagement metrics like likes count.
* Perform feature engineering with sentiment scores.
* Evaluate models using MAE, MSE, and R².
* Generate future predictions for sample inputs.

---
## 🛠 Technologies Used
* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn
* TextBlob

---
## 🔗 Dataset
The dataset was sourced from **NASA Climate Change Facebook comments (2020-2023)**, containing:
* Date of comment
* Likes count
* Comments count
* Text content
* Anonymised profile names

---
## 📊 Key Features Implemented
✅ Data cleaning and missing value handling
✅ Sentiment analysis using TextBlob
✅ Feature engineering (sentiment polarity + labels)
✅ Data visualisation (sentiment distribution plot)
✅ Regression modeling: Random Forest & Linear Regression
✅ Model evaluation: MAE, MSE, R² score
✅ Future predictions sample
✅ Final processed dataset saved as CSV

---

## 🚀 How to Run
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/climate-change-modeling.git
   cd climate-change-modeling
   ```
2. Install required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Ensure your dataset file is named **climate_nasa.csv** in the working directory.
4. Run the script:
   ```
   python main.py
   ```

---
## 📂 Output Files
* **processed_climate_nasa.csv** – Cleaned dataset with sentiment features
* **sentiment_distribution.png** – Sentiment label distribution plot

---
## ✨ Future Scope
* Advanced NLP topic modeling to extract dominant discussion themes
* Deployment as a Flask web app for live climate change comment sentiment analysis
* Integrate time series forecasting for long-term climate impact studies

---
