
# Data Science Project: Market Data Generation & Predictive Modeling

## Overview

This project presents a two-phase data science pipeline focused on synthetic feature generation and stock market movement prediction. By integrating language model prompting techniques and traditional machine learning workflows, the project explores a novel approach to financial data augmentation and classification modeling.

## Project Explanation

This project leverages ChatGPT for synthetic feature generation through prompt engineering. 

The algorithm begins iterating through a dataset created by combining various API Calls to the Polygon API holding various information about a stock.

On each phase of iteration the algorithm then selects from a wide variety of  prompt introduction prefixes for each prompt as well as changing the “temperature” a preset parameter for the output of the model, this is useful for enhancing the quality and uniqueness of each feature.

The next step that the algorithm uses is to add an output line. This line helps structure the output of each prompt, making it more easily compatible with tabular datasets that it later needs to be refined into.

Finally the algorithm interjects a core prompt, which will dynamically read through the Polygon dataset and include both the target stock ticker and date of interest.

Finally the algorithm sends the prompt to the ChatGPT API for a response, and will repeat this process several times for each different “core prompt” in the array. 

After all the iterations are done the result are several synthetic features which can then used as input for machine learning models aimed at analyzing and forecasting stock price behavior.

---

## Contents

### `Data Gen.ipynb`
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

### `Data Analysis.ipynb`
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
