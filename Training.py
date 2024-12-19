import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Model import save_model

class AmesHousingTraining:
    def __init__(self, data_path, target, random_seed=42):
        self.data_path = data_path
        self.target = target
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_seed
        )

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def preprocess(self, df):
        # Drop unnecessary columns
        cols_to_drop = ['Order', 'PID', 'Alley', 'Fence', 'Pool QC', 'Misc Feature', 'Mo Sold']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Feature engineering
        df['Surface_area'] = df['1st Flr SF'] + df['2nd Flr SF'] + df['Total Bsmt SF']
        df['Age_sold'] = df['Yr Sold'] - df['Year Built']
        df['Quality_and_Area'] = df['Overall Qual'] * df['Gr Liv Area']

        # Drop 'Yr Sold' after feature engineering
        df = df.drop(columns=['Yr Sold'], errors='ignore')

        # Separate features and target
        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Handle missing values
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X[categorical_cols] = X[categorical_cols].fillna('Unknown')

        # Scale numeric columns
        scaler = MinMaxScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # One-hot encode categorical columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Save feature names for consistency
        self.feature_names = list(X.columns)
        np.save('feature_names.npy', self.feature_names, allow_pickle=True)

        return X, y

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train the model and evaluate on the test set.
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))



    def save_split_data(self, X_train, X_test, y_train, y_test):
        """
        Save the train-test split as .npy files for consistency.
        """
        np.save('train_features.npy', X_train, allow_pickle=True)
        np.save('test_features.npy', X_test, allow_pickle=True)
        np.save('train_labels.npy', y_train, allow_pickle=True)
        np.save('test_labels.npy', y_test, allow_pickle=True)

    def run(self):
        """
        Main method to run the training pipeline.
        """
        df = self.load_data()
        X, y = self.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

        self.train_and_evaluate(X_train, X_test, y_train, y_test)
        self.save_split_data(X_train, X_test, y_train, y_test)
        save_model(self.model, 'trained_model.pkl')


if __name__ == "__main__":
    DATA_PATH = 'AmesHousing.csv'
    TARGET = 'SalePrice'

    trainer = AmesHousingTraining(DATA_PATH, TARGET)
    trainer.run()
