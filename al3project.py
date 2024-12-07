import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class AmesHousingPrediction:
    """
    A class to encapsulate the process of predicting Ames Housing prices 
    with a preliminary implementation for Milestone 2.
    """

    def __init__(self, data_path, target):
        """
        Initialize the class with dataset path and target variable.
        """
        self.data_path = data_path
        self.target = target
        self.selected_features = [
            'Overall Qual', 'Gr Liv Area', 'Neighborhood',
            'Year Built', 'Garage Area', 'Full Bath',
            'Bedroom AbvGr', 'TotRms AbvGrd'
        ]
        self.numeric_cols = None
        self.categorical_cols = None
        self.scaler = None
        self.encoder = None
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # Increased n_estimators

    def load_data(self):
        """
        Load the dataset and filter for selected features.
        """
        df = pd.read_csv(self.data_path)
        df = df[self.selected_features + [self.target]]
        return df

    def explore_data(self, df):
        """
        Perform exploratory data analysis on the dataset with basic visualizations.
        """
        print("\nDataset Info:")
        print(df.info())
        print("\nSummary Statistics:")
        print(df.describe())
        print("\nMissing Values Count:")
        print(df.isnull().sum())

        # Visualizing missing data (only missing features)
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            print(f"\nMissing data:\n{missing_data}")
            # Bar plot for missing values
            plt.figure(figsize=(10, 6))
            missing_data.plot(kind='bar', color='orange')
            plt.title("Missing Values in Features")
            plt.xlabel("Features")
            plt.ylabel("Count of Missing Values")
            plt.show()

    def feature_engineering(self, df):
        """
        Perform correlation analysis (basic) and log it.
        """
        corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
        print("\nCorrelation Matrix with Target Variable (SalePrice):")
        if self.target in corr_matrix:
            print(corr_matrix[self.target].sort_values(ascending=False))

        # Visualize correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap with SalePrice")
        plt.show()

    def preprocess(self, df):
        """
        Preprocess the dataset by handling missing values, scaling, and encoding.
        """
        # Separate the target
        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Determine numeric and categorical columns AFTER dropping the target
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns

        # Handle missing values
        X[self.numeric_cols] = X[self.numeric_cols].fillna(X[self.numeric_cols].median())
        X[self.categorical_cols] = X[self.categorical_cols].fillna('Unknown')

        # Scale numeric features
        self.scaler = StandardScaler()
        X_numeric = self.scaler.fit_transform(X[self.numeric_cols])

        # Encode categorical features (using sparse_output=False for a dense array)
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_categorical = self.encoder.fit_transform(X[self.categorical_cols])

        # Combine numeric and categorical features
        X_preprocessed = np.hstack([X_numeric, X_categorical])

        return X_preprocessed, y

    def train_and_evaluate(self, X, y):
        """
        Train the model and evaluate its performance on a test set.
        """
        # Split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate custom accuracy (within 10% of actual sale price)
        tolerance = 0.10
        relative_errors = np.abs((y_pred - y_test) / y_test)
        accurate_predictions = np.sum(relative_errors < tolerance)
        custom_accuracy = (accurate_predictions / len(y_test)) * 100

        # Display performance metrics (basic output)
        print("\nPreliminary Model Performance:")
        print(f"Custom Accuracy (within 10% of actual price): {custom_accuracy:.2f}%")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

    def run_experiment(self):
        """
        Execute the workflow from loading data to evaluation.
        """
        print("Loading data...")
        df = self.load_data()

        print("Exploring data...")
        self.explore_data(df)

        print("Feature engineering...")
        self.feature_engineering(df)

        print("Preprocessing data...")
        X, y = self.preprocess(df)

        print("Training and evaluating the model...")
        self.train_and_evaluate(X, y)


if __name__ == "__main__":
    # Define the path to the dataset and the target variable
    DATA_PATH = 'AmesHousing.csv'  # Update with your actual file path
    TARGET = 'SalePrice'

    # Initialize and run the experiment
    ames_housing = AmesHousingPrediction(DATA_PATH, TARGET)
    ames_housing.run_experiment()
