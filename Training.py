import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Model import save_model

class AmesHousingTraining:
    def __init__(self, data_path, target, random_seed=42):
        self.data_path = data_path
        self.target = target
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.selected_features = [
            'Overall Qual', 'Gr Liv Area', 'Neighborhood',
            'Year Built', 'Garage Area', 'Full Bath',
            'Bedroom AbvGr', 'TotRms AbvGrd', 'Lot Frontage', 'Lot Area', 'Street'
        ]
        self.model = RandomForestRegressor(n_estimators=200, random_state=self.random_seed)

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df[self.selected_features + [self.target]]
        return df

    def feature_engineering(self, df):
        """
        Perform feature correlation analysis and visualization.
        """
        corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
        print("\nCorrelation Matrix with Target Variable:")
        print(corr_matrix[self.target].sort_values(ascending=False))

        # Heatmap visualization
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap with Target Variable")
        plt.show()

    def explore_data(self, df):
        """
        Perform exploratory data analysis on the dataset.
        """
        print("\nDataset Info:")
        print(df.info())
        print("\nSummary Statistics:")
        print(df.describe())
        print("\nMissing Values Count:")
        print(df.isnull().sum())

        # SalePrice distribution visualization
        plt.figure(figsize=(8, 6))
        sns.histplot(df[self.target], kde=True, color='blue', bins=30)
        plt.title(f"Distribution of {self.target}")
        plt.xlabel(self.target)
        plt.ylabel("Frequency")
        plt.show()

    def preprocess(self, df):
        X = df.drop(columns=[self.target])
        y = df[self.target]

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X[categorical_cols] = X[categorical_cols].fillna('Unknown')

        X_preprocessed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        return X_preprocessed, y


    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Train the model and evaluate on the test set.
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Residual plot visualization
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color='red', bins=30)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

        # Display feature importance
        self.display_feature_importance(feature_names)

    def display_feature_importance(self, feature_names):
        """
        Visualize feature importances from the trained model.
        """
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        # Plot the top 10 features with a color palette
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis', hue='Feature', legend=False)
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.show()

    
    def save_split_data(self, X_train, X_test, y_train, y_test):
        """
        Save the train-test split and feature names as .npy files for consistency.
        """
        np.save('train_features.npy', X_train, allow_pickle=True)
        np.save('test_features.npy', X_test, allow_pickle=True)
        np.save('train_labels.npy', y_train, allow_pickle=True)
        np.save('test_labels.npy', y_test, allow_pickle=True)

        # Save feature names
        feature_names = X_train.columns
        np.save('feature_names.npy', feature_names, allow_pickle=True)

        print("Train-test split and feature names saved as .npy files.")

    def run(self):
        """
        Main method to run the training pipeline.
        """
        df = self.load_data()
        # Feature correlation analysis
        self.feature_engineering(df)

        # Perform exploratory data analysis
        self.explore_data(df)

        # Preprocess data
        X, y = self.preprocess(df)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

        # Train and evaluate the model
        feature_names = list(X.columns)
        self.train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

        # Save split data
        self.save_split_data(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    DATA_PATH = 'AmesHousing.csv'
    TARGET = 'SalePrice'
    MODEL_PATH = 'trained_model.pkl'

    trainer = AmesHousingTraining(DATA_PATH, TARGET)
    trainer.run()

    save_model(trainer.model, MODEL_PATH)
