import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the level of logger
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()  # console handler
f_handler = logging.FileHandler('logfile.log')  # file handler
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


class CreditCardFraudDetection:
    def __init__(self, filepath):
        """
        Class constructor. Initializes the file path and loads the data into a pandas dataframe.

        Parameters:
        filepath (str): Path to the .csv file containing the data.
        """
        self.filepath = filepath
        try:
            self.data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}")
        except Exception as e:
            logger.error("Error occurred while loading data.")
            logger.exception(e)

    def explore_data(self):
        """
        Explore the data by printing various statistics and plots.
        """
        try:
            # Print the first few rows of the data
            logger.info("First few rows of the data:")
            print(self.data.head())
            print()

            # Print the data types and non-null counts of the columns
            logger.info("Data types and non-null counts of the columns:")
            print(self.data.info())
            print()

            # Print the number of missing values in each column
            logger.info("Number of missing values in each column:")
            print(self.data.isnull().sum())
            print()

            # Print the number of unique values in each column
            logger.info("Number of unique values in each column:")
            print(self.data.nunique())
            print()

            # Print the correlation matrix
            logger.info("Correlation matrix:")
            print(self.data.corr())
            print()

            # Print some basic statistics of the numerical variables
            logger.info("Basic statistics of the numerical variables:")
            print(self.data.describe())
            print()

            # Plot histograms for each variable
            logger.info("Plotting histograms for each variable:")
            self.data.hist(figsize=(20, 20))
            plt.show()
        except Exception as e:
            logger.error("Error occurred while exploring data.")
            logger.exception(e)


    def visualize_data(self):
        """
        Visualize the class distributions.
        """
        try:
            sns.countplot(x='Class', data=self.data)
            plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
            plt.show()
        except Exception as e:
            logger.error("Error occurred while visualizing data.")
            logger.exception(e)

    def handle_imbalance(self):
        """
        Handle class imbalance using SMOTE.
        """
        try:
            X = self.data.drop('Class', axis=1)
            y = self.data['Class']

            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)

            # Convert resampled data back to dataframes, maintaining column names
            X_res = pd.DataFrame(X_res, columns=X.columns)
            y_res = pd.Series(y_res, name=y.name)

            return X_res, y_res
        except Exception as e:
            logger.error("Error occurred while handling imbalance.")
            logger.exception(e)



    def split_data(self, X, y):
        """
        Split the data into training and testing sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        """
        Scale the features using StandardScaler.
        """
        # Define the feature names
        feature_names = X_train.columns.tolist()
        
        # Create a scaler object
        scaler = StandardScaler()
        
        # Fit the scaler to the training data and transform it
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        
        # Use the fitted scaler to transform the test data
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

        return X_train_scaled, X_test_scaled

    def build_evaluate_models(self, X_train, y_train, X_test, y_test):
        """
        Build and evaluate models using cross-validation and compute feature importance.
        """
        # Define models
        models = [
            ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
            ('RandomForest', RandomForestClassifier(random_state=42)),
            ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ]

        # Cross-validation and performance comparison
        for name, model in models:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            print(f'{name} AUC: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

            # Fit the model and make predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate the model
            print(classification_report(y_test, y_pred))

            # Compute and plot Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall curve for {name}')
            plt.show()

            # Compute Area Under the Precision-Recall Curve (AUPRC)
            print('AUPRC:', auc(recall, precision))

            # Feature importance 
            if name == 'RandomForest':
                # Get feature importances
                importances = model.feature_importances_
                features = X_train.columns  # get feature names from the training data

                # Create a DataFrame for visualization
                feature_importances = pd.DataFrame({'feature': features, 'importance': importances})

                # Sort by importance
                feature_importances = feature_importances.sort_values('importance', ascending=False)

                # Plot feature importances
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=feature_importances)
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.show()


# usage:

try:
    detector = CreditCardFraudDetection('creditcard.csv')
    detector.explore_data()
    detector.visualize_data()

    X, y = detector.handle_imbalance()
    X_train, X_test, y_train, y_test = detector.split_data(X, y)
    X_train, X_test = detector.scale_data(X_train, X_test)
    detector.build_evaluate_models(X_train, y_train, X_test, y_test)
except Exception as e:
    logger.error("Error occurred while detecting fraud.")
    logger.exception(e)
