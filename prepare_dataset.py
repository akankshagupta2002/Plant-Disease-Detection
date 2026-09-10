from pathlib import Path
import random
import json
import shutil

# =========================
# SETTINGS
# =========================

SOURCE_DIR = Path("dataset/color")
OUTPUT_DIR = Path("dataset/split")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

# =========================
# CHECK SETTINGS
# =========================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 0.001:
    raise ValueError("Train/Validation/Test ratios must add up to 1.")

random.seed(SEED)

# =========================
# FIND CLASSES
# =========================

classes = sorted(
    [folder for folder in SOURCE_DIR.iterdir() if folder.is_dir()]
)

print(f"Found {len(classes)} classes.")

# =========================
# CREATE OUTPUT FOLDERS
# =========================

for split in ["train", "val", "test"]:
    for class_folder in classes:
        (OUTPUT_DIR / split / class_folder.name).mkdir(
            parents=True,
            exist_ok=True
        )

# =========================
# SPLIT DATA
# =========================

total_images = 0

for class_folder in classes:

    images = [
        file for file in class_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copy files
    for file in train_images:
        shutil.copy2(
            file,
            OUTPUT_DIR / "train" / class_folder.name / file.name
        )

    for file in val_images:
        shutil.copy2(
            file,
            OUTPUT_DIR / "val" / class_folder.name / file.name
        )

    for file in test_images:
        shutil.copy2(
            file,
            OUTPUT_DIR / "test" / class_folder.name / file.name
        )

    total_images += total

    print(
        f"{class_folder.name}: "
        f"{len(train_images)} train | "
        f"{len(val_images)} val | "
        f"{len(test_images)} test"
    )

# =========================
# SAVE CLASS NAMES
# =========================

with open(OUTPUT_DIR / "class_names.json", "w") as f:
    json.dump([c.name for c in classes], f, indent=4)

# =========================
# FINAL INFORMATION
# =========================

print("\n==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")

print(f"Total classes: {len(classes)}")
print(f"Total images: {total_images}")

print("\nSplit:")
print("Train: 80%")
print("Validation: 10%")
print("Test: 10%")

print(f"\nOutput folder: {OUTPUT_DIR}")

print("\nDataset split successful! ✅")