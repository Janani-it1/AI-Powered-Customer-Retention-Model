
## AI Powered Customer Retention Model
## Project Overview

This project demonstrates how to use machine learning to predict customer churn based on a dataset from a telecom company. The goal of this project is to build a predictive model that can identify customers who are likely to leave the service, allowing the company to take proactive measures to retain them. The project covers everything from data preprocessing to model training and evaluation.
## Dataset


The dataset used for this project is the Telco Customer Churn dataset, available from Kaggle. It includes information such as:

Customer demographics (gender, age, etc.)
Service usage (internet service, contract type, etc.)
Billing information (monthly charges, total charges)
Churn label (whether the customer left the service or not)

# Dataset File:

data/Telco_Churn_Dataset.csv

## Workflow


1. Data Preprocessing
The dataset undergoes cleaning and preprocessing:

Handling missing values: Missing data in the TotalCharges column is filled with the column mean.
Encoding categorical variables: Features such as gender, contract, and others are converted into numerical form using Label Encoding.
Feature scaling: Numerical features like tenure and MonthlyCharges are scaled using StandardScaler to normalize them.
Refer to the notebooks/Data_Preprocessing.ipynb notebook or src/data_processing.py script for detailed preprocessing steps.

2. Model Training
We train a Random Forest Classifier to predict whether a customer will churn. Other models, like Logistic Regression and XGBoost, can also be added for comparison.

Train-Test Split: We split the data into training and testing sets.
Model Training: The Random Forest model is trained on the processed data.
Feature Importance: The model identifies which features are the most important for predicting customer churn.
Refer to notebooks/Model_Training.ipynb or src/model_training.py for more details.

3. Model Evaluation
After training the model, we evaluate its performance using:

Accuracy: The percentage of correct predictions.
Precision, Recall, and F1-Score: Metrics to evaluate the model's performance on each class (churn/no churn).
Confusion Matrix: A breakdown of correct and incorrect predictions.
Refer to notebooks/Model_Training.ipynb or src/model_evaluation.py for detailed evaluation results.

# Model Performance
Here are the key performance metrics from the Random Forest Classifier model:

Accuracy: ~79%
Precision: Measures how many churn predictions were correct.
Recall: Measures how many actual churners were correctly identified.
F1-Score: Balances precision and recall for a comprehensive performance metric.
Feature Importance
The model identifies the most important features contributing to churn prediction, such as:

Tenure (how long the customer has been with the company),
Monthly Charges,
Contract Type (month-to-month contracts often have higher churn rates).
#Output of the Code

The output of the project, after following the steps and running the code, will be a comprehensive machine learning model that predicts customer churn. Here’s what the output looks like at different stages of the project:

1. Data Preprocessing (Output)
The data will be cleaned and processed. Missing values in the TotalCharges column will be filled, and categorical columns (such as gender, contract type, etc.) will be encoded into numerical values.
Example (after preprocessing):


customerID    gender   SeniorCitizen   Partner   Dependents   tenure  ...
7590-VHVEG     0           0            1         0           1      ...
5575-GNVDE     1           0            0         0          34      ...
This prepares the data for model training.

2. Model Training (Output)
After training the Random Forest Classifier (or any other model), you will receive an accuracy score indicating how well the model performed on the test set.
Example Output:


Accuracy: 0.792
This means that the model correctly predicts customer churn about 79.2% of the time.

3. Model Evaluation (Output)
The evaluation will give you detailed metrics on how the model performs on predicting churners (Class 1) vs non-churners (Class 0). The output will look like this:

Accuracy:
makefile
Copy code
Accuracy: 0.79 (79%)
Classification Report:

              precision    recall  f1-score   support

       0       0.83      0.91      0.87      1036
       1       0.65      0.47      0.54       373

    accuracy                           0.79      1409
   macro avg       0.74      0.69      0.70      1409
weighted avg       0.78      0.79      0.78      1409
Interpretation:

Precision for Class 0 (Non-Churn): 83% of customers predicted as not churning are correctly classified.
Recall for Class 1 (Churn): The model correctly identifies 47% of the actual churners.
F1-Score for Churn: The harmonic mean of precision and recall for churn prediction is 0.54.

4. Confusion Matrix (Optional Output)
The confusion matrix shows the breakdown of correct vs incorrect predictions:


Confusion Matrix:
[[941  95]   # 941 true negatives (correct non-churn), 95 false positives (incorrect churn predictions)
 [198 175]]  # 198 false negatives (missed churns), 175 true positives (correct churn predictions)

5. Feature Importance (Output)
You can visualize or print the importance of different features in predicting customer churn.

Example Output:

Feature Importances:
['tenure', 'MonthlyCharges', 'Contract', 'TotalCharges', ...]
This output will show which features (e.g., tenure, contract type) have the biggest impact on the model’s churn prediction.