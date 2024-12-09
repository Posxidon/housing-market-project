import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Model import save_model

class AmesHousingPrediction:
    """
    Predicting Ames Housing prices 
    """
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
        print("\nDataset Info:")
        print(df.info())
        print("\nSummary Statistics:")
        print(df.describe())
        print("\nMissing Values Count:")
        print(df.isnull().sum())

        # Visualizing missing data
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            print(f"\nMissing data summary:\n{missing_data}")

    def feature_engineering(self, df):
        corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
        print("\nCorrelation Matrix with Target Variable (SalePrice):")
        if self.target in corr_matrix:
            print(corr_matrix[self.target].sort_values(ascending=False))

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap with SalePrice")
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.histplot(df[self.target], kde=True, color='blue', bins=30)
        plt.title("Distribution of SalePrice")
        plt.xlabel("SalePrice")
        plt.ylabel("Frequency")
        plt.show()

    def preprocess(self, df):
        X = df.drop(columns=[self.target])
        y = df[self.target]

        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns

        X[self.numeric_cols] = X[self.numeric_cols].fillna(X[self.numeric_cols].median())
        X[self.categorical_cols] = X[self.categorical_cols].fillna('Unknown')

        self.scaler = StandardScaler()
        X_numeric = self.scaler.fit_transform(X[self.numeric_cols])

        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_categorical = self.encoder.fit_transform(X[self.categorical_cols])

        X_preprocessed = np.hstack([X_numeric, X_categorical])
        return X_preprocessed, y

    def display_feature_importance(self):
        importances = self.model.feature_importances_
        feature_names = list(self.numeric_cols) + list(self.encoder.get_feature_names_out(self.categorical_cols))
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.show()

    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    def run_experiment(self):
        df = self.load_data()
        self.explore_data(df)
        self.feature_engineering(df)
        X, y = self.preprocess(df)
        self.train_and_evaluate(X, y)
        self.display_feature_importance()

if __name__ == "__main__":
    DATA_PATH = 'AmesHousing.csv'
    TARGET = 'SalePrice'
    MODEL_PATH = 'trained_model.pkl'

    ames_housing = AmesHousingPrediction(DATA_PATH, TARGET)
    ames_housing.run_experiment()
    save_model(ames_housing.model, MODEL_PATH)
