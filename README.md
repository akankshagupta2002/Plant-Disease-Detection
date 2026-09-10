# Plant Disease Detection Using Deep Learning

## 1. Project Overview

This project implements a deep-learning-based plant disease classification
system using the PlantVillage color-image dataset and MobileNetV2 transfer
learning.

The model classifies plant leaf images into 38 different healthy/disease
categories and provides a predicted class along with a confidence score.

---

## 2. Dataset

Dataset: PlantVillage Color Images

- Total images: 54,305
- Number of classes: 38
- Image size used by the model: 224 × 224 pixels
- Image type: RGB/Color images

The dataset was divided into:

- Training: 80%
- Validation: 10%
- Testing: 10%

---

## 3. Model

The project uses MobileNetV2 with ImageNet pretrained weights.

Architecture:

Input Image
    ↓
224 × 224 × 3
    ↓
MobileNetV2
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Softmax Layer
    ↓
38 Classes

---

## 4. Training Configuration

- Model: MobileNetV2
- Input size: 224 × 224
- Batch size: 32
- Maximum epochs: 10
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Sparse Categorical Crossentropy
- Output classes: 38

The best validation accuracy was obtained at Epoch 9.

Best validation accuracy: 95.53%

---

## 5. Test Results

The final model was evaluated on the unseen test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 95.11% |
| Macro Precision | 94.96% |
| Macro Recall | 93.11% |
| Macro F1-score | 93.71% |
| Weighted Precision | 95.37% |
| Weighted Recall | 95.11% |
| Weighted F1-score | 95.03% |
| Test Images | 5,459 |

---

## 6. Class-wise Performance

The project generates detailed class-wise metrics in:

results/classification_report.csv

and:

results/per_class_performance.csv

Best performing class:

Cherry (including sour) — healthy

F1-score: 1.0000

Most challenging class:

Tomato — Early blight

F1-score: 0.6839

---

## 7. Evaluation Outputs

The following files are generated in the `results` directory:

- accuracy_curve.png
- loss_curve.png
- confusion_matrix.png
- classification_report.csv
- per_class_performance.csv
- evaluation_summary.json
- training_history.json

---

## 8. Saved Model

The trained models are stored in:

models/

Files:

- plant_disease_mobilenetv2.keras
- plant_disease_final.keras
- class_names.json

The `.keras` file contains the trained neural-network model.

---

## 9. Running the Project

### Step 1 — Activate virtual environment

Windows PowerShell:

    .\venv\Scripts\Activate.ps1

If PowerShell gives an execution-policy issue, use Command Prompt:

    venv\Scripts\activate

---

### Step 2 — Prepare the dataset

Run:

    python prepare_dataset.py

This prepares the dataset and creates the train/validation/test split.

---

### Step 3 — Train the model

Run:

    python train_model.py

The trained model will be saved in the `models` directory.

---

### Step 4 — Evaluate the model

Run:

    python evaluate_model.py

This generates:

- Classification report
- Confusion matrix
- Per-class performance
- Evaluation summary

---

### Step 5 — Generate sample predictions

Run:

    python predict_samples.py

The script generates sample prediction images and saves the prediction
information in:

    sample_predictions/sample_predictions.csv

Each prediction contains:

- Actual label
- Predicted label
- Confidence score
- Correct/Incorrect result

---

### Step 6 — Generate training graphs

Run:

    python create_results.py

This generates:

- results/accuracy_curve.png
- results/loss_curve.png

---

## 10. Sample Inference Results

The project evaluates 12 selected test images.

Result:

- Total samples: 12
- Correct: 12
- Incorrect: 0
- Sample accuracy: 100%

Important:

The 100% accuracy refers only to the selected 12 samples.
The overall test-set accuracy is 95.11%.

---

## 11. Project Structure

    Plant-Disease-Detection/
    │
    ├── dataset/
    │   ├── color/
    │   └── split/
    │
    ├── models/
    │   ├── class_names.json
    │   ├── plant_disease_mobilenetv2.keras
    │   └── plant_disease_final.keras
    │
    ├── results/
    │   ├── accuracy_curve.png
    │   ├── loss_curve.png
    │   ├── confusion_matrix.png
    │   ├── classification_report.csv
    │   ├── per_class_performance.csv
    │   ├── evaluation_summary.json
    │   └── training_history.json
    │
    ├── sample_predictions/
    │   ├── prediction_01.png
    │   ├── ...
    │   ├── prediction_12.png
    │   └── sample_predictions.csv
    │
    ├── prepare_dataset.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── predict_samples.py
    ├── create_results.py
    └── README.md

---

## 12. Conclusion

The project successfully demonstrates plant disease classification using
MobileNetV2 transfer learning.

The final model achieved 95.11% accuracy on the unseen test dataset across
38 plant disease/healthy categories.

The project also provides class-wise evaluation, confusion matrix,
training curves, confidence-based sample predictions and a saved model
artifact for future inference.