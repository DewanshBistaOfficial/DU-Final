
# Data Science Project: Market Data Generation & Predictive Modeling

## Overview

This project is a two-part data science workflow focused on generating synthetic market data and performing predictive analysis to determine the direction of market movements. It includes data collection and transformation, exploratory analysis, and the implementation of machine learning models for classification.

---

## Contents

### 📁 `Data Gen.ipynb`
This notebook is responsible for generating and formatting synthetic market data, using APIs and randomization techniques.

**Key Features:**
- Generates synthetic or fetched financial data.
- Prepares the data for downstream analysis.
- Uses APIs such as Polygon and OpenAI for data enrichment.

**Main Libraries:**
- `pandas`
- `openai`
- `polygon`
- `datetime`
- `random`
- `os`

---

### 📁 `Data Analysis.ipynb`
This notebook focuses on analyzing the generated data and applying machine learning techniques to predict market behavior.

**Main Tasks:**
- Data transformation for binary classification (e.g., price up or down).
- Feature correlation and collinearity analysis.
- Data visualization using Seaborn and Matplotlib.
- Model training using:
  - Logistic Regression
  - Random Forest Classifier
- Evaluation using accuracy and classification reports.

**Main Libraries:**
- `pandas`, `seaborn`, `matplotlib`
- `sklearn` (for model training and evaluation)
- `statsmodels` (for collinearity diagnostics)

---

## How to Use

1. **Generate Data**  
   Run `Data Gen.ipynb` to create or fetch the dataset.

2. **Analyze and Predict**  
   Open `Data Analysis.ipynb` to perform exploratory analysis and model training.

---

## Requirements

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

**`requirements.txt` content:**
```
pandas
scikit-learn
statsmodels
seaborn
matplotlib
openai
polygon-api-client
```

---

## Author

Dewansh Bista  
🔗 [LinkedIn](https://www.linkedin.com/in/dewansh-bista/)
