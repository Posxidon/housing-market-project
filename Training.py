import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from Model import save_model

class AmesHousingTraining:
    def __init__(self, data_path, target, random_seed=42):
        self.data_path = data_path
        self.target = target
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.model = RandomForestRegressor(random_state=self.random_seed)
        self.best_model = None  # Placeholder for the tuned model

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def feature_engineering(self, df):
        # Correlation matrix for feature selection
        corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
        print("\nCorrelation Matrix with Target Variable:")
        print(corr_matrix[self.target].sort_values(ascending=False))

        top_features = corr_matrix[self.target].sort_values(ascending=False).head(15).index
        top_corr_matrix = corr_matrix.loc[top_features, top_features]

        plt.figure(figsize=(10, 8))
        sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Heatmap of Top Correlated Features with SalePrice")
        plt.show()

    def explore_data(self, df):
        print("\nDataset Info:")
        print(df.info())
        print("\nSummary Statistics:")
        print(df.describe())
        print("\nMissing Values Count:")
        print(df.isnull().sum())

        plt.figure(figsize=(8, 6))
        sns.histplot(df[self.target], kde=True, color='blue', bins=30)
        plt.title(f"Distribution of {self.target}")
        plt.xlabel(self.target)
        plt.ylabel("Frequency")
        plt.show()

    def preprocess(self, df):
        # Drop unnecessary columns
        cols_to_drop = ['Order', 'PID', 'Alley', 'Fence', 'Pool QC', 'Misc Feature', 'Mo Sold']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Feature engineering
        df['Surface_area'] = df['1st Flr SF'] + df['2nd Flr SF'] + df['Total Bsmt SF']
        df['Age_sold'] = df['Yr Sold'] - df['Year Built']
        df['Quality_and_Area'] = df['Overall Qual'] * df['Gr Liv Area']
        df = df.drop(columns=['Yr Sold'], errors='ignore')

        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Handle missing values and ensure data types
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Fill missing numeric values with median and categorical with 'Unknown'
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X[categorical_cols] = X[categorical_cols].fillna('Unknown')

        # Ensure that MinMaxScaler only applies to numeric columns
        scaler = MinMaxScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # One-hot encoding for categorical columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        self.feature_names = list(X.columns)
        np.save('feature_names.npy', self.feature_names, allow_pickle=True)

        return X, y

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color='red', bins=30)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

        self.display_feature_importance(self.feature_names)

    def display_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis', hue='Importance')
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.show()

    def grid_search(self, X, y):
        param_grid = {
            'n_estimators': [50, 100, 150],  
            'max_depth': [5, 10, 15, None],  
            'max_features': ['sqrt', 'log2', 0.5],  
            'min_samples_split': [2, 5], 
            'min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=2, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
        grid_search.fit(X, y)
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Score: {-grid_search.best_score_:.2f}")
        self.best_model = grid_search.best_estimator_

    def plot_learning_curves(self, X, y):
        train_sizes, train_scores, test_scores = learning_curve(
            self.best_model if self.best_model else self.model,
            X, y, cv=2, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )

        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label='Training Error', color='blue')
        plt.plot(train_sizes, test_scores_mean, label='Validation Error', color='orange')
        plt.title("Learning Curves")
        plt.xlabel("Training Size")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.show()

    def save_split_data(self, X_train, X_test, y_train, y_test):
        np.save('train_features.npy', X_train, allow_pickle=True)
        np.save('test_features.npy', X_test, allow_pickle=True)
        np.save('train_labels.npy', y_train, allow_pickle=True)
        np.save('test_labels.npy', y_test, allow_pickle=True)

    def run(self):
        df = self.load_data()
        self.feature_engineering(df)
        self.explore_data(df)

        X, y = self.preprocess(df)

        # Perform grid search and cross-validation
        self.grid_search(X, y)

        # Plot learning curves
        self.plot_learning_curves(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
        self.train_and_evaluate(X_train, X_test, y_train, y_test)
        self.save_split_data(X_train, X_test, y_train, y_test)
        save_model(self.best_model if self.best_model else self.model, 'trained_model.pkl')


if __name__ == "__main__":
    DATA_PATH = 'AmesHousing.csv'
    TARGET = 'SalePrice'

    trainer = AmesHousingTraining(DATA_PATH, TARGET)
    trainer.run()
