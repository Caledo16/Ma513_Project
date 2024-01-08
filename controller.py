from train import train_model, evaluate_model
from tensorflow.keras.saving import load_model

if __name__ == '__main__':
    model, test_data, test_label = train_model()
    evaluate_model(model, test_data, test_label)



