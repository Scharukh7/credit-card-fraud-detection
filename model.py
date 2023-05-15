import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc

class CreditCardFraudDetection:
    def __init__(self, filepath):
        """
        Class constructor. Initializes the file path and loads the data into a pandas dataframe.
        
        Parameters:
        filepath (str): Path to the .csv file containing the data.
        """
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

    def explore_data(self):
        """
        Explore the data by printing various statistics and plots.
        """
        # Print the first few rows of the data
        print("First few rows of the data:")
        print(self.data.head())
        print()

        # Print the data types and non-null counts of the columns
        print("Data types and non-null counts of the columns:")
        print(self.data.info())
        print()

        # Print the number of missing values in each column
        print("Number of missing values in each column:")
        print(self.data.isnull().sum())
        print()

        # Print the number of unique values in each column
        print("Number of unique values in each column:")
        print(self.data.nunique())
        print()

        # Print the correlation matrix
        print("Correlation matrix:")
        print(self.data.corr())
        print()

        # Print some basic statistics of the numerical variables
        print("Basic statistics of the numerical variables:")
        print(self.data.describe())
        print()

        # Plot histograms for each variable
        print("Histograms for each variable:")
        self.data.hist(figsize=(20, 20))
        plt.show()


    def visualize_data(self):
        """
        Visualize the class distributions.
        """
        sns.countplot(x='Class', data=self.data)
        plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
        plt.show()

    def handle_imbalance(self):
        """
        Handle class imbalance using SMOTE.
        """
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        return X_res, y_res

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
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test

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
            model.fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            print(f'{name} AUC: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

            # Make predictions
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
                features = self.data.columns[:-1]  # assuming 'Class' is the last column

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

detector = CreditCardFraudDetection('creditcard.csv')
detector.explore_data()
detector.visualize_data()

X, y = detector.handle_imbalance()
X_train, X_test, y_train, y_test = detector.split_data(X, y)
X_train, X_test = detector.scale_data(X_train, X_test)
detector.build_evaluate_models(X_train, y_train, X_test, y_test)
