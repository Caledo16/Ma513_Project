import numpy as np
from tensorflow.keras.utils import to_categorical


def load_data(file_path):
    """
    Load data from an .npz file.

    Parameters:
    file_path (str): Path to the .npz file containing the data.

    Returns:
    data: Raw data from the .npz file.
    """
    # Load dataset
    data = np.load(file_path, allow_pickle=True)
    return data


def preprocess_data(data):
    """
    Preprocess the data.

    Parameters:
    data: Raw data from the .npz file.

    Returns:
    X (np.array): Preprocessed image data.
    y (np.array): Labels.
    """
    # Extract images and labels
    images_labels = data['arr']

    X = np.array([entry[0] for entry in images_labels])
    y = np.array([entry[1] for entry in images_labels])


    X_normalized = np.array([img / 255.0 for img in X])

    # Reshape images to include the channel dimension
    X_reshaped = np.expand_dims(X_normalized, axis=-1)

    # Convert labels to one-hot encoding
    y = to_categorical(y)

    return X_reshaped, y
