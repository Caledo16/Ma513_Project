from cnn_model import build_cnn_model
from load_data import load_data, preprocess_data
from sklearn.model_selection import train_test_split


def train_model():
    """

    :return:
    """
    file_path = 'images_malware.npz'
    data = load_data(file_path)
    X, y = preprocess_data(data)

    # Split the data into training, testing and validation sets
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)

    # Adjust the shape according to your dataset
    input_shape = train_data.shape[1:]

    model = build_cnn_model(input_shape)
    model.fit(train_data, train_labels,epochs=50, validation_data=(val_data, val_labels))

    model.save('malware_detection_model.h5')

    return model, test_data, test_labels


def evaluate_model(model, test_data, test_labels):
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print("Test accuracy: ", test_accuracy)