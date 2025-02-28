# Module_20_Challenge : Credit Risk Classification

This repository features a supervised machine learning model using logistic regression for credit risk classification. The analysis is processed via Python in a Jupyter Notebook.

## Repository Directory
|Folder|File|Description|
|---|---|---|
|Resources|lending_data.csv|Data file|
|root|credit_risk_classification.ipynb|Python Jupyter Notebook for analysis|

# Analysis
## Overview
The purpose of this analysis is to classify loans as "Healthy" or "High-Risk" based on financial factors using a logistic regression machine learning model.

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
+ A Logistic Regression model is fit with the training data.
+ The testing data is then run through the model to determine the model's reliability for correctly classifying new data points.

## Results
+ Very high overall accuracy (99.4%)
+ Healthy Loans: very high precision (99.8%) and recall (99.5%)
+ High-Risk Loans: fairly high precision (87.3%), high recall (94.9%)

## Summary
