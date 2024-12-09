import pickle

def save_model(model, file_path="trained_model_al3_group51.pkl"):
    """
    Save the trained model to a pickle file.

    Parameters:
    - model: The trained model object to be saved.
    - file_path

    Returns:
    None
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")


def load_model(file_path="trained_model_al3_group51.pkl"):
    """
    Load a trained model from a pickle file.

    Parameters:
    - file_path

    Returns:
    - The loaded model object.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
