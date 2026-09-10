from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================

TEST_DIR = Path("dataset/split/test")
MODEL_PATH = Path("models/plant_disease_mobilenetv2.keras")
OUTPUT_DIR = Path("sample_predictions")

IMG_SIZE = (224, 224)
NUM_SAMPLES = 12
SEED = 42

OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)

# =========================
# CHECK FILES
# =========================

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# =========================
# LOAD MODEL
# =========================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully! ✅")

# =========================
# LOAD CLASS NAMES
# =========================

class_names_path = Path("models/class_names.json")

if class_names_path.exists():
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
else:
    class_names = sorted(
        [folder.name for folder in TEST_DIR.iterdir() if folder.is_dir()]
    )

print(f"Number of classes: {len(class_names)}")

# =========================
# COLLECT TEST IMAGES
# =========================

image_files = []

for class_folder in TEST_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    for image_file in class_folder.iterdir():

        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_files.append(image_file)

if len(image_files) < NUM_SAMPLES:
    NUM_SAMPLES = len(image_files)

# Randomly select samples
samples = random.sample(image_files, NUM_SAMPLES)

print(f"\nSelected {NUM_SAMPLES} sample images.")

# =========================
# PREDICTIONS
# =========================

results = []

for index, image_path in enumerate(samples, start=1):

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    prediction = model.predict(
        image_batch,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(prediction))

    predicted_class = class_names[predicted_index]

    confidence = float(
        prediction[predicted_index] * 100
    )

    # Actual class comes from folder name
    actual_class = image_path.parent.name

    correct = predicted_class == actual_class

    results.append({
        "image": image_path.name,
        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence, 2),
        "correct": correct
    })

    # =========================
    # SAVE VISUAL RESULT
    # =========================

    plt.figure(figsize=(8, 6))

    plt.imshow(image)
    plt.axis("off")

    title = (
        f"Actual: {actual_class}\n"
        f"Predicted: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )

    plt.title(title, fontsize=10)

    output_path = OUTPUT_DIR / f"prediction_{index:02d}.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"{index:02d}. "
        f"Actual: {actual_class} | "
        f"Predicted: {predicted_class} | "
        f"Confidence: {confidence:.2f}% | "
        f"{'CORRECT' if correct else 'WRONG'}"
    )

# =========================
# SAVE CSV
# =========================

results_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "sample_predictions.csv"

results_df.to_csv(
    csv_path,
    index=False
)

# =========================
# SUMMARY
# =========================

correct_count = int(results_df["correct"].sum())
total_count = len(results_df)

print("\n==============================")
print("SAMPLE PREDICTIONS COMPLETE")
print("==============================")

print(f"Total samples: {total_count}")
print(f"Correct: {correct_count}")
print(f"Incorrect: {total_count - correct_count}")

print(
    f"Sample accuracy: "
    f"{correct_count / total_count * 100:.2f}%"
)

print(f"\nImages saved to: {OUTPUT_DIR}")
print(f"CSV saved to: {csv_path}")

print("\nDone! ✅")
