from pathlib import Path
import json

import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================

RESULTS_DIR = Path("results")
HISTORY_FILE = RESULTS_DIR / "training_history.json"

if not HISTORY_FILE.exists():
    raise FileNotFoundError(
        f"Training history not found: {HISTORY_FILE}"
    )

# =========================
# LOAD HISTORY
# =========================

with open(HISTORY_FILE, "r") as f:
    history = json.load(f)

accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(1, len(accuracy) + 1)

# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True)

accuracy_path = RESULTS_DIR / "accuracy_curve.png"

plt.tight_layout()
plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# LOSS GRAPH
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True)

loss_path = RESULTS_DIR / "loss_curve.png"

plt.tight_layout()
plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# BEST RESULTS
# =========================

best_epoch = val_accuracy.index(max(val_accuracy)) + 1
best_val_accuracy = max(val_accuracy)

print("\n==============================")
print("RESULT GRAPHS CREATED")
print("==============================")

print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")

print(f"\nAccuracy graph: {accuracy_path}")
print(f"Loss graph: {loss_path}")

print("\nResults generation complete! ✅")