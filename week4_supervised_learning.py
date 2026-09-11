# ============================================
# WEEK 4 - SUPERVISED LEARNING
# Titanic Survival Prediction
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------

df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# --------------------------------------------
# 2. INITIAL DATA CHECK
# --------------------------------------------

print("\nMissing Values Before Preprocessing:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Survived"].value_counts())

print("\nTarget Distribution (%):")
print(df["Survived"].value_counts(normalize=True) * 100)

# --------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------

# Family size = siblings/spouse + parents/children + passenger
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# IsAlone = 1 if passenger travelled alone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Extract title from passenger name
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)

# Group rare titles
common_titles = ["Mr", "Miss", "Mrs", "Master"]

df["Title"] = df["Title"].apply(
    lambda x: x if x in common_titles else "Rare"
)

print("\nFeature Engineering Completed.")

print("\nNew Features:")
print(df[["FamilySize", "IsAlone", "Title"]].head())

# --------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# --------------------------------------------

X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title"
    ]
]

y = df["Survived"]

# --------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# --------------------------------------------
# 6. PREPROCESSING
# --------------------------------------------

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"
]

categorical_features = [
    "Sex",
    "Embarked",
    "Title"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# --------------------------------------------
# 7. LOGISTIC REGRESSION
# --------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

logistic_model.fit(X_train, y_train)

print("\nLogistic Regression training completed.")

# --------------------------------------------
# 8. RANDOM FOREST
# --------------------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight=None
            )
        )
    ]
)
random_forest_model.fit(X_train, y_train)

print("Random Forest training completed.")

# --------------------------------------------
# 9. PREDICTIONS
# --------------------------------------------

logistic_predictions = logistic_model.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)

print("\nPredictions generated successfully.")

# --------------------------------------------
# 10. BASIC ACCURACY
# --------------------------------------------

from sklearn.metrics import accuracy_score

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\n====================================")
print("MODEL ACCURACY")
print("====================================")

print(
    f"Logistic Regression Accuracy: "
    f"{logistic_accuracy * 100:.2f}%"
)

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)
# --------------------------------------------
# 11. DETAILED MODEL EVALUATION
# --------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Logistic Regression metrics
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

logistic_precision = precision_score(y_test, logistic_predictions)
logistic_recall = recall_score(y_test, logistic_predictions)
logistic_f1 = f1_score(y_test, logistic_predictions)
logistic_auc = roc_auc_score(y_test, logistic_probabilities)

# Random Forest metrics
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
rf_auc = roc_auc_score(y_test, rf_probabilities)

# --------------------------------------------
# 12. PRINT MODEL COMPARISON
# --------------------------------------------

print("\n====================================")
print("DETAILED MODEL COMPARISON")
print("====================================")

print("\nLogistic Regression:")
print(f"Accuracy  : {logistic_accuracy:.4f}")
print(f"Precision : {logistic_precision:.4f}")
print(f"Recall    : {logistic_recall:.4f}")
print(f"F1-Score  : {logistic_f1:.4f}")
print(f"ROC-AUC   : {logistic_auc:.4f}")

print("\nRandom Forest:")
print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1-Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_auc:.4f}")

# --------------------------------------------
# 13. CONFUSION MATRICES
# --------------------------------------------

print("\n====================================")
print("CONFUSION MATRICES")
print("====================================")

print("\nLogistic Regression:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nRandom Forest:")
print(confusion_matrix(y_test, rf_predictions))

# --------------------------------------------
# 14. CLASSIFICATION REPORTS
# --------------------------------------------

print("\n====================================")
print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("====================================")

print(classification_report(y_test, logistic_predictions))

print("\n====================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("====================================")

print(classification_report(y_test, rf_predictions))
# --------------------------------------------
# 15. CROSS-VALIDATION
# --------------------------------------------

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_cv_scores = cross_val_score(
    logistic_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

rf_cv_scores = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n====================================")
print("5-FOLD CROSS-VALIDATION")
print("====================================")

print("\nLogistic Regression CV Scores:")
print(logistic_cv_scores)

print(
    f"Mean Accuracy: "
    f"{logistic_cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: "
    f"{logistic_cv_scores.std():.4f}"
)

print("\nRandom Forest CV Scores:")
print(rf_cv_scores)

print(
    f"Mean Accuracy: "
    f"{rf_cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: "
    f"{rf_cv_scores.std():.4f}"
)
# --------------------------------------------
# 16. HYPERPARAMETER TUNING - LOGISTIC REGRESSION
# --------------------------------------------

from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    logistic_model,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n====================================")
print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
print("====================================")

print("Best Parameters:")
print(grid_search.best_params_)

print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# Evaluate tuned model on test set
tuned_logistic_predictions = grid_search.predict(X_test)

tuned_logistic_accuracy = accuracy_score(
    y_test,
    tuned_logistic_predictions
)

print(
    f"Tuned Logistic Regression Test Accuracy: "
    f"{tuned_logistic_accuracy:.4f}"
)
# --------------------------------------------
# 17. LOGISTIC REGRESSION FEATURE INTERPRETATION
# --------------------------------------------

# Get the fitted preprocessing pipeline
fitted_preprocessor = grid_search.best_estimator_.named_steps["preprocessor"]

# Get transformed feature names
feature_names = fitted_preprocessor.get_feature_names_out()

# Get logistic regression coefficients
coefficients = (
    grid_search.best_estimator_
    .named_steps["classifier"]
    .coef_[0]
)

# Create feature importance table
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients)
})

# Sort by absolute coefficient
feature_importance = feature_importance.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\n====================================")
print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
print("====================================")

print(
    feature_importance[
        ["Feature", "Coefficient"]
    ].head(15)
)
# --------------------------------------------
# 18. FEATURE IMPORTANCE VISUALIZATION
# --------------------------------------------

top_features = feature_importance.head(10).sort_values(
    by="Coefficient"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 10 Logistic Regression Features")
plt.tight_layout()

plt.savefig("logistic_feature_importance.png", dpi=300)
plt.show()

print("\nFeature importance graph saved as logistic_feature_importance.png")
# --------------------------------------------
# 19. MODEL PERFORMANCE COMPARISON
# --------------------------------------------

models = ["Logistic Regression", "Random Forest"]
accuracies = [0.8436, 0.8101]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.02, f"{value:.2%}", ha="center")

plt.tight_layout()

plt.savefig("model_accuracy_comparison.png", dpi=300)
plt.show()

print("\nModel comparison graph saved as model_accuracy_comparison.png")
