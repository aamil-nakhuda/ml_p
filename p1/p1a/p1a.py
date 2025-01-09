# Difficult code
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# # Load the dataset from a CSV file
# file_path = 'data.csv'  # Replace with your file path
# df = pd.read_csv(file_path)
# # Display the first few rows
# print(df.head())
# # Handle Missing Values
# print("Missing values before cleaning:")

# print(df.isnull().sum())
# # Fill missing values with mean for numerical columns and mode for categorical columns
# df['numerical_column'] = df['numerical_column'].fillna(df['numerical_column'].mean())
# df['category_column'] = df['category_column'].fillna(df['category_column'].mode()[0])
# # Handle Inconsistent Formatting (Example: Strip spaces and convert to lowercase)
# df.columns = df.columns.str.strip()  # Strip spaces from column names
# df['category_column'] = df['category_column'].str.lower()  # Convert text to lowercase
# # Handle Outliers (Using IQR)
# Q1 = df['numerical_column'].quantile(0.25)
# Q3 = df['numerical_column'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# df_cleaned = df[(df['numerical_column'] >= lower_bound) & (df['numerical_column'] <= upper_bound)]
# # Display cleaned data
# print("Cleaned data:")
# print(df_cleaned.head())
# # Visualize with Boxplot to check for outliers
# sns.boxplot(x=df_cleaned['numerical_column'])
# plt.title('Boxplot of Numerical Column After Cleaning')
# plt.show()

# # Visualize the distribution with a histogram
# df_cleaned['numerical_column'].hist(bins=30)
# plt.title('Histogram of Numerical Column After Cleaning')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

# Easier Code
# import pandas as pd
# import numpy as np
# from scipy import stats
# # Load Titanic dataset (replace with your actual file path)
# df = pd.read_csv('p1/titanic.csv') # Adjust file name if necessary
# # Display first few rows to understand the structure
# print("Initial Data:")
# print(df.head())

# # 1. Handle Missing Values
# # For simplicity, we will fill missing values in numeric columns with the column mean.
# # For categorical columns, we will fill missing values with the mode (most frequent value).
# # Fill missing values in numeric columns with the mean
# df['Age'] = df['Age'].fillna(df['Age'].mean())
# df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
# # Fill missing values in categorical columns with the mode
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# df['Cabin'] = df['Cabin'].fillna('Unknown') # Fill Cabin with a default string if missing

# # 2. Handle Inconsistent Formatting
# # Convert 'Sex' column to lowercase to standardize the text
# df['Sex'] = df['Sex'].str.lower()
# # Normalize 'Embarked' column by stripping extra spaces and ensuring consistency in uppercase
# df['Embarked'] = df['Embarked'].str.strip().str.upper()

# # 3. Handle Outliers using Z-score method
# # We'll apply outlier detection on the numeric columns: 'Age' and 'Fare'
# numeric_columns = ['Age', 'Fare']
# # Calculate Z-scores for numeric columns to detect outliers
# z_scores = np.abs(stats.zscore(df[numeric_columns]))
# # Set a threshold for Z-scores to identify outliers (e.g., Z > 3)
# threshold = 3
# outliers = (z_scores > threshold)
# # Remove outliers by filtering out rows where any numeric column exceeds the threshold
# df_no_outliers = df[(z_scores < threshold).all(axis=1)]
# # Display cleaned data
# print("\nCleaned Data (after handling missing values, inconsistent formatting, and outliers):")
# print(df_no_outliers.head())
# # Save the cleaned dataset if needed
# df_no_outliers.to_csv('p1/cleaned_titanic_data.csv', index=False)

# Best and easiest code
import pandas as pd
import numpy as np

# Sample dataset
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Name': ['John', 'Mary', 'Bob', 'Alice', 'Charlie', np.nan, 'David', 'Emma', 'Frank', 'Grace'],
    'Age': [25, np.nan, 45, 35, 28, 38, 50, 30, 22, 70],
    'Income': [50000, 70000, 120000, np.nan, 85000, 95000, 300000, 45000, 52000, 220000],
    'Join_Date': ['2020-01-10', '2021-03-15', '2020-02-20', '2020-05-30', '2019-09-10',
                  '2021-12-25', '2020-10-01', '2018-07-18', '2022-04-05', '2020-01-15']
}

# Load dataset into DataFrame
df = pd.DataFrame(data)
# 1. Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing Age with median
df['Income'].fillna(df['Income'].median(), inplace=True)  # Fill missing Income with median
df['Name'].fillna('Unknown', inplace=True)  # Replace missing Name with 'Unknown'

# # 2. Handle inconsistent formatting
df['Join_Date'] = pd.to_datetime(df['Join_Date'], format='%Y-%m-%d')  # Standardize date format

# 3. Handle outliers (we can cap the 'Income' at a certain percentile to remove extreme outliers)
income_cap = df['Income'].quantile(0.95)  # Set the 95th percentile as the cap
df['Income'] = np.where(df['Income'] > income_cap, income_cap, df['Income'])  # Cap outliers

# Check the cleaned dataset
print(df.head)
