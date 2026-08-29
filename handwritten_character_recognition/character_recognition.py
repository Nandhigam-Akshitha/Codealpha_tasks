import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Task 3: Handwritten Character Recognition
# Dataset: MNIST
# Model: Convolutional Neural Network (CNN)
# ---------------------------------------------------------

print("TensorFlow version:", tf.__version__)
print("\nLoading MNIST dataset...")


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Add channel dimension for CNN
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


print("Training samples:", x_train.shape[0])
print("Testing samples:", x_test.shape[0])


# ---------------------------------------------------------
# Build CNN model
# ---------------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])


# Display model architecture
print("\nCNN Model:")
model.summary()


# ---------------------------------------------------------
# Compile model
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

print("\nTraining model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

print("\nEvaluating model on test data...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ---------------------------------------------------------
# Sample Predictions
# ---------------------------------------------------------

sample_predictions = model.predict(
    x_test[:10],
    verbose=0
)

sample_labels = np.argmax(
    sample_predictions,
    axis=1
)

print("\nSample Predictions:")

for i in range(10):
    print(
        f"Sample {i + 1}: "
        f"Predicted = {sample_labels[i]}, "
        f"Actual = {y_test[i]}"
    )
# ---------------------------------------------------------
# Predictions for the complete test dataset
# ---------------------------------------------------------

all_predictions = model.predict(
    x_test,
    verbose=0
)

predicted_labels = np.argmax(
    all_predictions,
    axis=1
)
# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        predicted_labels,
        digits=4
    )
)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    predicted_labels
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

fig, ax = plt.subplots(figsize=(8, 8))

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title("MNIST CNN - Confusion Matrix")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Training and Validation Accuracy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.show()


# ---------------------------------------------------------
# Training and Validation Loss
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------

model.save("handwritten_character_cnn.keras")

print("\nModel saved as: handwritten_character_cnn.keras")
print("\nTask 3 training and evaluation completed successfully!")