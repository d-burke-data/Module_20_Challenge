# Module_20_Challenge : *Credit Risk Classification*

This repository features a supervised machine learning model using logistic regression for credit risk classification. The analysis is processed via Python in a Jupyter Notebook.

## Repository Directory
|Folder|File|Description|
|---|---|---|
|Resources|lending_data.csv|Data file|
|root|credit_risk_classification.ipynb|Python Jupyter Notebook for analysis|

# Analysis
## Overview
The purpose of this analysis is to classify loans as "Healthy" or "High-Risk" based on financial factors using a logistic regression machine learning model. Logistic regression is used here to classify loans based on a binary condition (Healthy or High-Risk).

### Data
The dataset is composed of 77,537 records of the following fields:

|Column|Data Type|
|---|---|
|loan_size|float|
|interest_rate|float|
|borrower_income|float|
|debt_to_income|float|
|num_of_accounts|int|
|derogatory_marks|int|
|total_debt|float|
|loan_status|boolean|

`loan_status` is the target variable, designated as:

|Value|Definition|
|---|---|
|0|Healthy Loan|
|1|High-Risk Loan|

### Procedure
+ The data is loaded from file *lending_data.csv* into a Pandas DataFrame and separated into target variable `y` (`loan_status`) and features variable `X` (all other columns).
+ The dataset is then divided into training and testing portions (75%/25% split, respectively), stratified by `y`.
+ The training and testing features are each scaled using the Standard Scaler based on the training data.
+ A Logistic Regression model is fit with the training data.
+ The testing data is then run through the model to determine the model's reliability for correctly classifying new data points.

## Results
+ Very high overall accuracy (99.5%).
+ Healthy loans precision at 99.9% indicates only 0.1% of truly High-Risk loans will be misclassified as Healthy.
+ Healthy loans recall at 99.5% indicates only 0.5% of truly Healthy loans will be misclassified as High-Risk.
+ High-Risk loans recall at 97.8% indicates only 2.2% of truly High-Risk loans will be misclassified as Healthy.
+ High-Risk loans precision at 87.2% indicates 12.8% of those identified as High-Risk will actually be Healthy.

## Summary
Overall, the model is very accurate (accuracy = 99.5%), though that is to be expected as 96.7% of the loans are Healthy. For Healthy loans, the model has very high precision (99.9%, indicating only 0.1% of the loans classified as Healthy were actually High-Risk) and recall (99.5%, indicating only 0.5% of the Healthy loans were classified as High-Risk). But the primary focus of the model is not on identifying the Healthy loans, which make up the vast majority and are generally assumed to be true. Instead, the model attempts to identify High-Risk loans, and the most important metric is High-Risk recall, as a low value could be financially disastrous for the company. With High-Risk recall at 97.8%, only 2.2% of High-Risk loans were classified as Healthy. High-Risk precision fared the worst when to testing the model, at 87.2%, meaning 12.8% of loans classified as High-Risk were actually Healthy. This is acceptable, however, as it will result in some Healthy loans receiving extra scrutiny, as opposed to High-Risk loans being overlooked (recall). Given this information, I support the company's use of this model.
