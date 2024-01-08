from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn_model(input_shape):
    """
    Create the model for the project
    :param input_shape: the shape of the input
    :return: model : return the created model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),  # Use padding='same'
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Use padding='same'
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(25, activation='softmax')  # Assuming 25 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model