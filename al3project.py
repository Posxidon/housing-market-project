import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class AmesHousingPrediction:
    """
    Predicting Ames Housing prices 
    """
    def __init__(self, data_path, target, random_seed=42):
        """
        Initialize the class 
        """
        self.data_path = data_path
        self.target = target
        self.random_seed = random_seed
        np.random.seed(self.random_seed)  # Set seed for reproducibility
        self.selected_features = [
            'Overall Qual', 'Gr Liv Area', 'Neighborhood',
            'Year Built', 'Garage Area', 'Full Bath',
            'Bedroom AbvGr', 'TotRms AbvGrd', 'Lot Frontage', 'Lot Area', 'Street'
        ]
        self.numeric_cols = None
        self.categorical_cols = None
        self.scaler = None
        self.encoder = None
        self.model = RandomForestRegressor(n_estimators=200, random_state=self.random_seed)  

    def load_data(self):
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
            print(f"\nMissing data summary:\n{missing_data}")
            print("\nMissing values have been imputed during preprocessing.")

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

        # Visualize target distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(df[self.target], kde=True, color='blue', bins=30)
        plt.title("Distribution of SalePrice")
        plt.xlabel("SalePrice")
        plt.ylabel("Frequency")
        plt.show()

    def preprocess(self, df):
        """
        Preprocess the dataset by handling missing values, scaling & encoding.
        """
        # Separate the target
        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Determine numeric and categorical columns post dropping the columns
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns

        # Handle missing values
        X[self.numeric_cols] = X[self.numeric_cols].fillna(X[self.numeric_cols].median())
        X[self.categorical_cols] = X[self.categorical_cols].fillna('Unknown')

        # Scale  features
        self.scaler = StandardScaler()
        X_numeric = self.scaler.fit_transform(X[self.numeric_cols])

        # Encode categorical features 
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_categorical = self.encoder.fit_transform(X[self.categorical_cols])

        # combine numeric and categorical features
        X_preprocessed = np.hstack([X_numeric, X_categorical])

        return X_preprocessed, y

    def display_feature_importance(self):
        """
        Display feature importance from the Random Forest model.
        """
        importances = self.model.feature_importances_
        feature_names = list(self.numeric_cols) + list(self.encoder.get_feature_names_out(self.categorical_cols))
        
        # Combine and sort by importance
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        # Plot the top 10 features
        plt.figure(figsize=(10, 6))
        # Updated barplot call to address the deprecation warning
        ax = sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance.head(10),
            hue='Feature',  # Add hue parameter
            legend=False    # Disable legend since we don't need it
        )
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.show()


    def display_residuals(self, y_test, y_pred):
        """
        Plot residuals (actual - predicted values).
        """
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color='red', bins=30)
        plt.title("Residuals Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

    def cross_validate_model(self, X, y, cv_folds=5):
        """
        Perform cross-validation to evaluate the model's performance.
        """
        # Define the cross-validation strategy
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        
        # Evaluate the model using cross-validation
        scores = cross_val_score(self.model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
        
        # Convert negative MAE to positive for readability
        mae_scores = -scores
        
        # Calculate average and standard deviation of MAE
        mean_mae = np.mean(mae_scores)
        std_mae = np.std(mae_scores)
        
        # Display cross-validation results
        print(f"\nCross-Validation Results (using {cv_folds}-fold):")
        print(f"Mean MAE: {mean_mae:.2f}")
        print(f"Standard Deviation of MAE: {std_mae:.2f}")
        print(f"Individual MAE scores: {mae_scores}")

    def train_and_evaluate(self, X, y):
        """
        Train the model and evaluate its performance on a test set.
        """
        # Split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

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

        # Print output
        print("\nPreliminary Model Performance:")
        print(f"Custom Accuracy (within 10% of actual price): {custom_accuracy:.2f}%")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        # Display residuals
        self.display_residuals(y_test, y_pred)

    def run_experiment(self, cv_folds=5):
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

        self.cross_validate_model(X, y, cv_folds=cv_folds)
        self.display_feature_importance()


if __name__ == "__main__":
    # Define the path to the dataset and the target variable
    DATA_PATH = 'AmesHousing.csv' 
    TARGET = 'SalePrice'

    # Initialize and run the experiment
    ames_housing = AmesHousingPrediction(DATA_PATH, TARGET)
    ames_housing.run_experiment()

