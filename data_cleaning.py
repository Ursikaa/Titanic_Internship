import pandas as pd

# Load the Titanic dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# Handle missing values

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Cabin with "Unknown"
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Handle missing values

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Cabin with "Unknown"
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Check for outliers

numerical_columns = ["Age", "SibSp", "Parch", "Fare"]

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"\n{column} - Number of outliers: {len(outliers)}")
    # Data preprocessing

# Convert Sex into numerical values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert Embarked into numerical values
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Display preprocessed data
print("\nData After Preprocessing:")
print(df.head())

# Check data types
print("\nData Types After Preprocessing:")
print(df.dtypes)
# Data preprocessing

# Convert Sex into numerical values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert Embarked into numerical values
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Display preprocessed data
print("\nData After Preprocessing:")
print(df.head())

# Check data types
print("\nData Types After Preprocessing:")
print(df.dtypes)
# Save cleaned and preprocessed dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")