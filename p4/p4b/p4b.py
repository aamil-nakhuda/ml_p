# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # Step 1: Read training data from a CSV file
# # Let's assume the CSV file has feature columns and the target column is named 'class'.
# data = pd.read_csv('p4/p4b/data.csv')  # Update with your actual file path

# # Split the data into features (X) and target labels (y)
# X = data.drop(columns=['class'])  # All columns except 'class'
# y = data['class']  # The target column

# # Step 2: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 3: Initialize the k-NN classifier and train the model
# k = 3  # You can change k to any number for the number of neighbors
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(X_train, y_train)

# # Step 4: Predict the labels for the test set
# y_pred = knn.predict(X_test)

# # Step 5: Print correct and wrong predictions with sample data
# for i in range(len(X_test)):
#     sample = X_test.iloc[i]
#     true_label = y_test.iloc[i]
#     predicted_label = y_pred[i]
    
#     if true_label == predicted_label:
#         print(f"Correct: Sample {i+1} -> Features: {sample.values} | True label: {true_label} | Predicted: {predicted_label}")
#     else:
#         print(f"Wrong: Sample {i+1} -> Features: {sample.values} | True label: {true_label} | Predicted: {predicted_label}")

# # Step 6: Evaluate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# New Code
# Below is a Python implementation of the **k-Nearest Neighbors (k-NN)** algorithm. The code is designed to read training data, build a model, classify a test sample, and print both correct and incorrect predictions. It includes inline comments for clarity.

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Step 1: Define the k-NN function
def k_nearest_neighbors(train_data, train_labels, test_sample, k=3):
    """
    Predict the label of a test sample using the k-Nearest Neighbors algorithm.
    """
    # Calculate Euclidean distances between test_sample and all training data
    distances = np.sqrt(((train_data - test_sample) ** 2).sum(axis=1))
    
    # Find the indices of the k nearest neighbors
    k_indices = distances.argsort()[:k]
    
    # Extract the labels of the k nearest neighbors
    k_nearest_labels = train_labels[k_indices]
    
    # Return the most common label
    return Counter(k_nearest_labels).most_common(1)[0][0]

# Step 2: Generate or load a sample dataset
# Example: Simple dataset (you can replace this with actual data)
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Perform k-NN classification and evaluate results
k = 5  # Number of neighbors
correct_predictions = 0
wrong_predictions = 0

for i, test_sample in enumerate(X_test):
    prediction = k_nearest_neighbors(X_train, y_train, test_sample, k)
    actual = y_test[i]
    if prediction == actual:
        correct_predictions += 1
        print(f"Correct: Predicted={prediction}, Actual={actual}")
    else:
        wrong_predictions += 1
        print(f"Wrong: Predicted={prediction}, Actual={actual}")

# Step 4: Print final summary
print("\nSummary:")
print(f"Correct Predictions: {correct_predictions}")
print(f"Wrong Predictions: {wrong_predictions}")
print(f"Accuracy: {correct_predictions / len(y_test) * 100:.2f}%")

### Key Features:
# 1. **k-NN Function:** Implements the algorithm with Euclidean distance.
# 2. **Dataset:** Uses the Iris dataset from `sklearn` for demonstration.
# 3. **Evaluation:** Splits data into training and testing sets for independent evaluation.
# 4. **Output:** Prints predictions for each test sample and provides a summary of accuracy.

### How It Works:
# 1. **Distance Calculation:** Calculates the Euclidean distance between the test sample and each training data point.
# 2. **Neighbor Selection:** Selects `k` nearest neighbors based on the smallest distances.
# 3. **Majority Voting:** Determines the most common label among neighbors.

# This code is straightforward and easily extendable for other datasets or enhancements like normalization. Let me know if you have specific datasets or additional requirements!