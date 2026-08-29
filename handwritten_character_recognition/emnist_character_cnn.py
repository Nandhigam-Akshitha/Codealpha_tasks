import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# EMNIST HANDWRITTEN CHARACTER RECOGNITION USING CNN
# ============================================================

print("\n==============================================")
print("      EMNIST CNN CHARACTER RECOGNITION")
print("==============================================\n")

# ------------------------------------------------------------
# 1. Load EMNIST Balanced dataset
# ------------------------------------------------------------

print("Loading EMNIST Balanced dataset...")

(ds_train, ds_test), ds_info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

num_classes = ds_info.features["label"].num_classes

print("Number of classes:", num_classes)
print("Training examples:", ds_info.splits["train"].num_examples)
print("Testing examples:", ds_info.splits["test"].num_examples)


# ------------------------------------------------------------
# 2. Convert dataset to NumPy arrays
# ------------------------------------------------------------

print("\nPreparing data...")

x_train = []
y_train = []

for image, label in tfds.as_numpy(ds_train):
    x_train.append(image)
    y_train.append(label)

x_test = []
y_test = []

for image, label in tfds.as_numpy(ds_test):
    x_test.append(image)
    y_test.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

print("Original training shape:", x_train.shape)
print("Original testing shape:", x_test.shape)


# ------------------------------------------------------------
# 3. Normalize images
# ------------------------------------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Processed training shape:", x_train.shape)
print("Processed testing shape:", x_test.shape)


# ------------------------------------------------------------
# 4. Build CNN model
# ------------------------------------------------------------

print("\nBuilding CNN model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ------------------------------------------------------------
# 5. Compile model
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ------------------------------------------------------------
# 6. Train CNN
# ------------------------------------------------------------

print("\nStarting CNN training...\n")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)


# ------------------------------------------------------------
# 7. Evaluate model
# ------------------------------------------------------------

print("\n==============================================")
print("             MODEL EVALUATION")
print("==============================================")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nTest Loss     :", test_loss)
print("Test Accuracy :", test_accuracy)


# ------------------------------------------------------------
# 8. Save trained model
# ------------------------------------------------------------

model.save("handwritten_character_emnist_cnn.keras")

print("\nModel saved as:")
print("handwritten_character_emnist_cnn.keras")


# ------------------------------------------------------------
# 9. Training and validation accuracy graph
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("EMNIST CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "emnist_training_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 10. Training and validation loss graph
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("EMNIST CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "emnist_training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 11. Sample predictions
# ------------------------------------------------------------

print("\nGenerating sample predictions...")

sample_indices = np.random.choice(
    len(x_test),
    10,
    replace=False
)

sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]

predictions = model.predict(
    sample_images,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        sample_images[i].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"True: {sample_labels[i]}\n"
        f"Pred: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.suptitle("EMNIST CNN - Sample Predictions")

plt.tight_layout()

plt.savefig(
    "emnist_sample_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 12. Confusion Matrix
# ------------------------------------------------------------

print("\nGenerating confusion matrix...")

# Use the complete test set for predictions
y_pred_prob = model.predict(
    x_test,
    batch_size=128,
    verbose=1
)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(16, 14))

sns.heatmap(
    cm,
    annot=False,
    cmap="viridis",
    fmt="d"
)

plt.title("EMNIST CNN - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig(
    "emnist_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 13. Classification report
# ------------------------------------------------------------

print("\n==============================================")
print("          CLASSIFICATION REPORT")
print("==============================================\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 14. Final output
# ------------------------------------------------------------

print("\n==============================================")
print(" EMNIST TRAINING AND EVALUATION COMPLETED")
print("==============================================")

print("\nGenerated files:")

print("1. handwritten_character_emnist_cnn.keras")
print("2. emnist_training_validation_accuracy.png")
print("3. emnist_training_validation_loss.png")
print("4. emnist_sample_predictions.png")
print("5. emnist_confusion_matrix.png")

print("\nEMNIST task completed successfully!")