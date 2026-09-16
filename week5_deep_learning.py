import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target distribution
print("\nSurvival Distribution:")
print(df["Survived"].value_counts())

# Target percentage
print("\nSurvival Percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)
# ==========================================
# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# ==========================================

data = df.copy()

# Drop unnecessary columns
data.drop(["PassengerId", "Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing Age with median
data["Age"] = data["Age"].fillna(data["Age"].median())

# Fill missing Embarked with mode
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Create FamilySize
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

# Create IsAlone
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

# Extract Title from Name
data["Title"] = data["Name"].str.extract(
    r",\s*([^.]*)\.", expand=False
)

# Group uncommon titles
common_titles = ["Mr", "Miss", "Mrs", "Master"]

data["Title"] = data["Title"].apply(
    lambda x: x if x in common_titles else "Rare"
)

# Drop original Name
data.drop("Name", axis=1, inplace=True)

# Display cleaned dataset
print("\nCleaned Dataset:")
print(data.head())

# Check missing values
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Display final columns
print("\nFinal Columns:")
print(data.columns.tolist())
# ==========================================
# STEP 3: ENCODING, DATA SPLITTING & SCALING
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked", "Title"],
    drop_first=True,
    dtype=int
)

print("\nFeatures After Encoding:")
print(X.head())

print("\nFeature Columns:")
print(X.columns.tolist())

# First split: 80% training, 20% temporary test data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: temporary data into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData Split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nScaled Training Data Shape:", X_train.shape)
print("Scaled Validation Data Shape:", X_val.shape)
print("Scaled Testing Data Shape:", X_test.shape)
# ==========================================
# STEP 3: ENCODING, DATA SPLITTING & SCALING
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked", "Title"],
    drop_first=True,
    dtype=int
)

print("\nFeatures After Encoding:")
print(X.head())

print("\nFeature Columns:")
print(X.columns.tolist())

# First split: 80% training, 20% temporary test data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: temporary data into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData Split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nScaled Training Data Shape:", X_train.shape)
print("Scaled Validation Data Shape:", X_val.shape)
print("Scaled Testing Data Shape:", X_test.shape)
# ==========================================
# STEP 4: CLASS WEIGHTS & NEURAL NETWORK
# ==========================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------
# 1. Calculate class weights
# ------------------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass Weights:")
print(class_weights)

# ------------------------------------------
# 2. Build Neural Network
# ------------------------------------------

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.30),

    Dense(16, activation="relu"),
    Dropout(0.20),

    Dense(1, activation="sigmoid")
])

# ------------------------------------------
# 3. Compile model
# ------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display architecture
print("\nNeural Network Architecture:")
model.summary()

# ------------------------------------------
# 4. Early stopping
# ------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ------------------------------------------
# 5. Train model
# ------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining completed.")
# ============================================================
# FINAL MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

# Test predictions
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

# Evaluation metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall   : {test_recall:.4f}")
print(f"Test F1-Score : {test_f1:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Survived", "Survived"]
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"]
)

plt.title("Deep Learning Model - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig("deep_learning_confusion_matrix.png", dpi=300)
plt.show()

# ============================================================
# TRAINING & VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("deep_learning_accuracy.png", dpi=300)
plt.show()

# ============================================================
# TRAINING & VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("deep_learning_loss.png", dpi=300)
plt.show()

# ============================================================
# BEST VALIDATION RESULTS
# ============================================================

best_val_accuracy = max(history.history["val_accuracy"])
best_val_loss = min(history.history["val_loss"])

print("\nBest Validation Accuracy:",
      f"{best_val_accuracy:.4f}")

print("Best Validation Loss:",
      f"{best_val_loss:.4f}")

print("\nEvaluation completed successfully.")
