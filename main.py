from sklearn.datasets import load_iris 
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import numpy as np


def main():
    # load data
    X, y = load_data()

    # shuffle data
    X, y = shuffle(X, y)

    # split data before transformation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # transformation and return scaler
    scaler, X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)

    # model input and output arquitecture
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    # get model
    model = get_model(input_shape, output_shape)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2%}")

    # Make predictions from SCALED unseen data
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_test_classes = y_test.argmax(axis=1)

    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_classes, y_pred_classes))
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    # unseen data
    data = np.array([[5.1, 3.5, 1.4, 0.2], [4, 2.5, 0.7, 0.1]])

    # predictions from UNSCALED unseen data
    scale_predict(data, scaler, model)


def load_data():

    """
    load evidence and label data and returns as np.ndarray type
    """

    # load iris dataset
    iris = load_iris()
    evidence, label = iris.data, iris.target
    return (evidence, label)


def preprocess(X_train, X_test, y_train, y_test):

    """"
    scale and returns `X_train`, `X_test`
    encode and returns `y_train`, `y_test`
    returns scaler for futuro unscaled data
    """

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode the target labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return (scaler, X_train, X_test, y_train, y_test)


def get_model(input_shape, output_shape):

    """
    returns a sequential compiled model with:
        Input layer with `input_shape` units
        Hideen layer with 10 units
        Hideen layer with 10 units
        Output layer with `output_shape` units
    """

    # Create a neural network model using Sequential API
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def scale_predict(data, scaler, model):
    """"
    function to scale useen data and pass into model prediction
    """
    
    # Standardize the new data using the same scaler used on the training data
    scaled_data = scaler.transform(data)

    # Make predictions using the pre-trained model
    predictions = model.predict(scaled_data)

    # 'predictions' will give you the predicted probabilities for each class
    print()
    print("New Data")
    print("Predicted probabilities:")
    print(np.array([[f"{x:.2%}" for x in row] for row in predictions]))


if __name__ == "__main__":
    main()