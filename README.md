# Credit Card Fraud Detection

## Note: 

CSV file is not in this repo as the csv file was large and couldn't upload, here's the link to the dataset card: 

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

## Context:

Credit Card Fraud is a significant problem in the financial industry. This project aims to use machine learning techniques to identify fraudulent transactions. The script provided uses a dataset of credit card transactions and builds several machine learning models to detect the fraudulent ones.

## About The Script

The provided script is a Python program that uses popular libraries such as pandas, scikit-learn, XGBoost, and seaborn for data processing, machine learning, and data visualization.

The script is organized into a single class, `CreditCardFraudDetection`, that encapsulates all steps of the data processing and machine learning pipeline. These steps include:

1. **Data Loading and Exploration:** The script loads a dataset from a given file path, and performs a preliminary analysis of the data, including displaying basic statistics and histograms of the variables.

2. **Data Visualization:** The script visualizes the class distributions in the dataset using a bar plot.

3. **Handling Class Imbalance:** The script uses the SMOTE technique to handle class imbalance in the dataset.

4. **Data Splitting:** The script splits the data into training and testing sets.

5. **Data Scaling:** The script scales the features using the StandardScaler from scikit-learn.

6. **Model Building and Evaluation:** The script builds and evaluates three models: Logistic Regression, Random Forest, and XGBoost. The models are evaluated using cross-validation and the Area Under the ROC Curve (AUC) metric.

## Problem Context

The data contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

## Results

The script outputs the performance of each model in terms of the AUC score and a classification report that includes metrics such as precision, recall, and F1-score. For the Random Forest model, the script also outputs a plot of feature importances.

## Installation

To run this script, you need to have Python installed on your system. The script uses the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost

You can install these libraries using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```
## Usage
After installing the dependencies, you can run the script using the following command:

```sh
python ./path/model.py
Replace ./'path'/ with the path to the script.
```
Note: The script assumes that the dataset is in a file named creditcard.csv in the same directory as the script. If your dataset is in a different location, modify the file path accordingly in the script.

## Data Description

The dataset contains numerical input variables V1 to V28 which are the result of a PCA transformation. Features that have not been transformed with PCA are 'Time' and 'Amount'. 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. The 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Model Explanation

The script uses three different models for fraud detection:

- **Logistic Regression**: This is a simple yet effective linear model that is widely used for binary classification problems.

- **Random Forest**: This is an ensemble method that operates by constructing multiple decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

- **XGBoost**: This is a powerful gradient boosting framework that uses decision trees. It's known for its performance and efficiency.

## Evaluation

The models are evaluated using cross-validation and the Area Under the ROC Curve (AUC) metric. AUC is a popular metric for binary classification problems, and it measures the quality of the model's predictions regardless of the classification threshold. An AUC closer to 1 indicates better performance. For the Random Forest model, the script also outputs a plot of feature importances.

## Conclusion

This script provides a basic pipeline for credit card fraud detection. However, it can be extended in various ways to improve performance. For example, could try different methods for handling class imbalance, feature engineering to create new features, or try different machine learning models.

## Improvements for Better Results

While the current implementation provides a robust foundation for credit card fraud detection, there are several potential areas of improvement that could potentially enhance the model's performance:

1. **Feature Engineering:** The current model uses the features as they are. Creating new features based on existing ones, such as interaction features, could potentially improve the model's predictive power.

2. **Hyperparameter Tuning:** The models are currently using mostly default hyperparameters. Systematically tuning these through techniques such as grid search or randomized search could potentially improve performance.

3. **Ensemble Methods:** Using ensemble methods, such as stacking or boosting, could potentially improve the overall predictive performance by combining the strengths of different models.

4. **Use of Advanced Algorithms:** More advanced machine learning algorithms like Neural Networks, or even Deep Learning models, could potentially provide better results, given their ability to model complex non-linear relationships.

5. **Anomaly Detection Techniques:** As this is a fraud detection problem, anomaly detection techniques could be particularly suited, as they are designed to detect rare events.

6. **Regularization Techniques:** To prevent overfitting, applying regularization techniques such as L1 and L2 could help improve the generalization of the model.



