import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Model import load_model

class AmesHousingTesting:
    def __init__(self, model_path):
        """
        Initialize with the path to the trained model.
        """
        self.model = load_model(model_path)

    def load_features(self):
        """
        Load test features, labels, and feature names saved during training.
        """
        X_test = np.load('test_features.npy', allow_pickle=True)
        y_test = np.load('test_labels.npy', allow_pickle=True)
        feature_names = np.load('feature_names.npy', allow_pickle=True)
        return X_test, y_test, feature_names

    def align_features(self, X_test, feature_names):

        X_test_df = pd.DataFrame(X_test, columns=feature_names[:X_test.shape[1]])
        for col in feature_names:
            if col not in X_test_df.columns:
                # add missing columns with 0
                X_test_df[col] = 0  
        X_test_df = X_test_df[feature_names] 
        return X_test_df

    def predict(self, X_test, feature_names):
        """
        Perform predictions using the loaded model.
        """
        X_test_aligned = self.align_features(X_test, feature_names)
        predictions = self.model.predict(X_test_aligned)
        return predictions

    def evaluate(self, y_test, y_pred):
        """
        Evaluate the model performance using MAE, RMSE, and accuracy.
        """
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate custom accuracy (within 10% of the actual sale price)
        tolerance = 0.10
        relative_errors = np.abs((y_pred - y_test) / y_test)
        accurate_predictions = np.sum(relative_errors < tolerance)
        custom_accuracy = (accurate_predictions / len(y_test)) * 100

        return mae, rmse, custom_accuracy

if __name__ == "__main__":
    MODEL_PATH = 'trained_model.pkl'

    # Initialize tester
    tester = AmesHousingTesting(MODEL_PATH)

    # Load test features, labels, and feature names
    X_test, y_test, feature_names = tester.load_features()

    # Align features and make predictions
    predictions = tester.predict(X_test, feature_names)

    # Evaluate predictions
    mae, rmse, accuracy = tester.evaluate(y_test, predictions)

    # Print evaluation metrics
    print("\nModel Evaluation on Test Data:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Accuracy (within 10% tolerance): {accuracy:.2f}%")
