import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Titanic dataset
df = pd.read_csv("cleaned_titanic.csv")

# Basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

# Survival count
print("\nSex values:")
print(df["Sex"].unique())
print(df["Sex"].value_counts(dropna=False))
print("\nSurvival Count:")
print(df["Survived"].value_counts())

# Survival rate
print("\nSurvival Rate:")
print(df["Survived"].value_counts(normalize=True) * 100)

# -------------------------------
# Visualization 1: Survival Count
# -------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("survival_count.png")
plt.show()

# -----------------------------------------
# Visualization 2: Survival by Sex
# -----------------------------------------

# Load original Titanic data for Sex analysis
sex_df = pd.read_csv("train.csv")

# Create a clear bar chart
plt.figure(figsize=(7, 5))

sns.countplot(
    data=sex_df,
    x="Sex",
    hue="Survived"
)

plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["Not Survived", "Survived"])
plt.tight_layout()

# Save graph
plt.savefig("survival_by_sex.png")

# Display graph
plt.show()


# -----------------------------------
# Visualization 3: Survival by Class
# -----------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()

# -------------------------------
# Visualization 4: Age Distribution
# -------------------------------
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Age", bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# --------------------------------
# Visualization 5: Fare Distribution
# --------------------------------
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Fare", bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("fare_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
print("\nEDA completed successfully!")
