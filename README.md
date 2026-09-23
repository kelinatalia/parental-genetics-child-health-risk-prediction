# Parental Genetics Child Health Risk Prediction

## Overview
This project predicts a child's health risk level (High, Moderate, or Low) based on parental genetic and physical data such as age, height, blood group, eye color, hair color, skin tone, and family disease history. The project covers the full pipeline from data exploration to model deployment as a live API, plus a simple web app to test it.

## Steps
- Data exploration: checked missing values, duplicates, and class balance for each feature
- Data preprocessing: filled missing values in family disease history, removed unused ID column, encoded categorical features, and scaled numeric features
- Modeling: trained and compared Random Forest, Logistic Regression, and XGBoost
- Evaluation: compared the 3 models using precision, recall, F1-score, and accuracy
- Deployment: packaged the trained model and deployed it as a live endpoint on AWS SageMaker
- Web app: built a Streamlit app so users can input data and get a prediction from the deployed endpoint directly

## Result
XGBoost was chosen as the final model for deployment based on its evaluation results.

## Tech Stack
Python, pandas, numpy, scikit-learn, XGBoost, AWS SageMaker, boto3, Streamlit
