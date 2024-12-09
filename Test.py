import pandas as pd
import numpy as np
from Model import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class TestAmesHousing:
    def __init__(self, model_path, data_path, target):
        self.model_path = model_path
        self.data_path = data_path
        self.target = target
        self.model = load_model(self.model_path)

    def preprocess(self, df):
        X = df.drop(columns=[self.target])
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X[categorical_cols] = X[categorical_cols].fillna('Unknown')

        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X[numeric_cols])

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_categorical = encoder.fit_transform(X[categorical_cols])

        X_preprocessed = np.hstack([X_numeric, X_categorical])
        return X_preprocessed

    def predict(self):
        df = pd.read_csv(self.data_path)
        X = self.preprocess(df)
        predictions = self.model.predict(X)
        return predictions

if __name__ == "__main__":
    TEST_DATA_PATH = 'AmesHousing_test.csv'
    MODEL_PATH = 'trained_model.pkl'
    TARGET = 'SalePrice'

    tester = TestAmesHousing(MODEL_PATH, TEST_DATA_PATH, TARGET)
    predictions = tester.predict()

    print(f"Predictions: {predictions[:5]}")
