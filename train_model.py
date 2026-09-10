from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# SETTINGS
# =========================

DATA_DIR = Path("dataset/split")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS = 10

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# =========================
# CHECK DATASET
# =========================

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

# =========================
# LOAD DATA
# =========================

print("Loading training data...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\nNumber of classes: {num_classes}")

# =========================
# SAVE CLASS NAMES
# =========================

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

# =========================
# PERFORMANCE
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================
# BASE MODEL
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# BUILD MODEL
# =========================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# =========================
# COMPILE
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# MODEL SUMMARY
# =========================

print("\n==============================")
print("MODEL SUMMARY")
print("==============================")

model.summary()

# =========================
# CALLBACKS
# =========================

best_model_path = MODEL_DIR / "plant_disease_mobilenetv2.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )
]

# =========================
# TRAIN
# =========================

print("\n==============================")
print("STARTING TRAINING")
print("==============================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# SAVE FINAL MODEL
# =========================

final_model_path = MODEL_DIR / "plant_disease_final.keras"

model.save(final_model_path)

# =========================
# TEST EVALUATION
# =========================

print("\n==============================")
print("TEST EVALUATION")
print("==============================")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# =========================
# SAVE TRAINING HISTORY
# =========================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(RESULTS_DIR / "training_history.json", "w") as f:
    json.dump(history_data, f, indent=4)

print("\n==============================")
print("TRAINING COMPLETE! ✅")
print("==============================")

print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
print(f"History: {RESULTS_DIR / 'training_history.json'}")