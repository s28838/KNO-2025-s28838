import sys
import os
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = "model.keras"

# Load or train model
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Training new model...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save learning curve
    with open("learning_curve.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.history.keys())
        writer.writerows(zip(*history.history.values()))
    print("Learning curve saved to learning_curve.csv")

# If image filename provided, predict digit
if len(sys.argv) > 1:
    image_path = sys.argv[1]

    if image_path == "--plot":
        # Plot learning curve
        with open("learning_curve.csv", "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)

        epochs = range(1, len(data) + 1)
        loss = [float(row["loss"]) for row in data]
        val_loss = [float(row["val_loss"]) for row in data]
        accuracy = [float(row["accuracy"]) for row in data]
        val_accuracy = [float(row["val_accuracy"]) for row in data]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label="Training Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label="Training Accuracy")
        plt.plot(epochs, val_accuracy, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        # Load and preprocess image
        img = Image.open(image_path).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        digit = np.argmax(predictions[0])

        print(f"Predicted digit: {digit}")
