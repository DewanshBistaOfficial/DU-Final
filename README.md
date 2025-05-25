
# Data Science Project: Market Data Generation & Predictive Modeling

## Overview

This project presents a two-phase data science pipeline focused on synthetic feature generation and stock market movement prediction. By integrating language model prompting techniques and traditional machine learning workflows, the project explores a novel approach to financial data augmentation and classification modeling.

## Project Explanation

This project leverages ChatGPT for synthetic feature generation through prompt engineering. 

The algorithm begins iterating through a dataset created by combining various API Calls to the Polygon API holding various information about a stock.

On each phase of iteration the algorithm then selects from a wide variety of prompt introduction prefixes for each prompt as well as changing the “temperature” a preset parameter for the output of the model, this is useful for enhancing the quality and uniqueness of each feature.

The next step that the algorithm uses is to add an output line. This line helps structure the output of each prompt, making it more easily compatible with tabular datasets that it later needs to be refined into.

Finally the algorithm interjects a core prompt, which will dynamically read through the Polygon dataset and include both the target stock ticker and date of interest.

Finally the algorithm sends the prompt to the ChatGPT API for a response, and will repeat this process several times for each different “core prompt” in the array. 

After all the iterations are done the result are several synthetic features which can then used as input for machine learning models aimed at analyzing and forecasting stock price behavior.

An example of contextual prefixes shown below:
![image](https://github.com/user-attachments/assets/ac154c55-a6df-4108-8593-caf042acbed7)

## Analysis And Issues

When generating the data, similar prompts could lead to collinearity, this was solved by adjusting the temperature, as well as changing prompt introductions

Another challenge is dealing with ‘hallucinations’ since GPT has no verification mechanism for it’s answers, the trick was to not ask it factual questions but more qualitative ones

Since GPT does not answer questions in a standardized format, there were issues extracting the exact answers needed from the response text this was partially solved by adding in a formatting line

As shown below one of the biggest issues with the features created was the fact that they tended to overtly skew in one answer regardless of the “real” outcome of the feature

![image](https://github.com/user-attachments/assets/6b7ef96d-ef09-4012-adfd-1b56adabb2b4)
![image](https://github.com/user-attachments/assets/864d70f2-af62-4616-bd25-16bcf64cf260)
![image](https://github.com/user-attachments/assets/c9e62546-1af0-413e-aa84-25e7f5d5d8dc)

## Models And Results

The models used to make the predictions were the Random Forest Algorithm and Logistic Regression, they were chosen due to their speed, as well as the small size of the dataset

These models are fairly interesting as both models did not perform well however, an accuracy of 50% suggest that the model is doing the equivalent of blindly guessing as the outcome has 2 options, this is somewhat to be expected when looking at earlier data and seeing how the outcomes skewed to one result in the generated feature, this is further corroborated when looking at the recall scores of both models, the extreme scores almost 0 and 1 respectively, suggesting something fishy with the given data

![image](https://github.com/user-attachments/assets/c7647384-014f-47bb-8a63-bdf4452aabe6)
![image](https://github.com/user-attachments/assets/f2ccb237-bf15-48e5-8ec2-cbcdbc461c14)

## Further Discussion

This project was limited to using GPT related data using different LLM systems, especially without content filters may have led to different results

Similarly the project only looked into tech stock, if looking at different security classes or stock types may have led to similarly different results

This project partially failed in objectives because the LLM data augmentation was used to replace what is normally complex human reccomendations, it seems that when using GPT generated data it may be better to stick to more simple questions

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

## Author

Dewansh Bista  
🔗 [LinkedIn](https://www.linkedin.com/in/dewansh-bista/)
