from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =========================
# SETTINGS
# =========================

TEST_DIR = Path("dataset/split/test")
MODEL_PATH = Path("models/plant_disease_mobilenetv2.keras")
RESULTS_DIR = Path("results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

RESULTS_DIR.mkdir(exist_ok=True)

# =========================
# CHECK FILES
# =========================

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# =========================
# LOAD TEST DATA
# =========================

print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print(f"Number of classes: {len(class_names)}")

# =========================
# LOAD MODEL
# =========================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully! ✅")

# =========================
# MAKE PREDICTIONS
# =========================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# OVERALL ACCURACY
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n==============================")
print("OVERALL PERFORMANCE")
print("==============================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

# Save report
report_path = RESULTS_DIR / "classification_report.csv"

report_df.to_csv(
    report_path,
    index=True
)

print("\nClassification Report:")
print(
    report_df.round(4).to_string()
)

print(
    f"\nClassification report saved to: {report_path}"
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(20, 18))

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Plant Disease Classification - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    rotation=90,
    fontsize=7
)

plt.yticks(
    rotation=0,
    fontsize=7
)

plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix saved to: {cm_path}"
)

# =========================
# PER-CLASS PERFORMANCE
# =========================

per_class = report_df.loc[
    class_names,
    ["precision", "recall", "f1-score", "support"]
]

per_class_path = RESULTS_DIR / "per_class_performance.csv"

per_class.to_csv(
    per_class_path
)

print(
    f"Per-class performance saved to: {per_class_path}"
)

# =========================
# BEST / WORST CLASSES
# =========================

best_class = per_class["f1-score"].idxmax()
worst_class = per_class["f1-score"].idxmin()

print("\n==============================")
print("CLASS PERFORMANCE")
print("==============================")

print(
    f"Best class: {best_class} "
    f"(F1 = {per_class.loc[best_class, 'f1-score']:.4f})"
)

print(
    f"Worst class: {worst_class} "
    f"(F1 = {per_class.loc[worst_class, 'f1-score']:.4f})"
)

# =========================
# SAVE SUMMARY
# =========================

summary = {
    "number_of_classes": len(class_names),
    "test_accuracy": float(accuracy),
    "best_class": best_class,
    "best_class_f1": float(
        per_class.loc[best_class, "f1-score"]
    ),
    "worst_class": worst_class,
    "worst_class_f1": float(
        per_class.loc[worst_class, "f1-score"]
    )
}

summary_path = RESULTS_DIR / "evaluation_summary.json"

with open(summary_path, "w") as f:
    json.dump(
        summary,
        f,
        indent=4
    )

# =========================
# COMPLETE
# =========================

print("\n==============================")
print("EVALUATION COMPLETE! ✅")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Report: {report_path}")
print(f"Confusion Matrix: {cm_path}")
print(f"Per-class Results: {per_class_path}")
print(f"Summary: {summary_path}")
