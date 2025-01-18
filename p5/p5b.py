# Skip this prac - Implement Hidden Markov Models using hmmlearn

# New code - still doesn't work
# import numpy as np
# from hmmlearn import hmm

# # Step 1: Prepare the data
# # Observations: 0 = No Umbrella, 1 = Umbrella
# observations = np.array([[0], [0], [1], [1], [0], [1], [0], [1]])

# # Step 2: Define and Train the HMM Model
# # Define the HMM: 2 hidden states (Sunny, Rainy), 2 possible observations (No Umbrella, Umbrella)
# model = hmm.MultinomialHMM(n_components=2, n_iter=1000, random_state=42, init_params='')  # Disable auto-initialization

# # Initialize the model's parameters
# # Start probabilities (Sunny = 0.5, Rainy = 0.5)
# model.startprob_ = np.array([0.5, 0.5])

# # Transition matrix (Sunny->Sunny = 0.7, Sunny->Rainy = 0.3, etc.)
# model.transmat_ = np.array([[0.7, 0.3],  # From Sunny
#                             [0.4, 0.6]])  # From Rainy

# # Emission probabilities (Sunny->No Umbrella = 0.8, Sunny->Umbrella = 0.2, etc.)
# model.emissionprob_ = np.array([[0.8, 0.2],  # From Sunny
#                                 [0.3, 0.7]])  # From Rainy
# # The shape of emissionprob_ is (2, 2), matching (n_components, n_features)

# # Train the model on the observations
# model.fit(observations)

# # Step 3: Predict the hidden states (weather) given the observations
# predicted_states = model.predict(observations)

# # Step 4: Output results
# print("Predicted Hidden States (Weather):")
# print(predicted_states)

# # Step 5: Output the model's parameters (transition matrix, emission matrix)
# print("\nTransition Matrix (State to State):")
# print(model.transmat_)
# print("\nEmission Matrix (Observation | State):")
# print(model.emissionprob_)

# # Step 6: Making predictions for new observations (example)
# # Let's predict the weather based on a new sequence of umbrella usage
# new_observations = np.array([[0], [1], [1]])  # No Umbrella, Umbrella, Umbrella
# predicted_new_states = model.predict(new_observations)
# print("\nPredicted Hidden States for New Observations (No Umbrella, Umbrella, Umbrella):")
# print(predicted_new_states)


# Old code
# import numpy as np
# from hmmlearn import hmm
# # Step 1: Prepare the data
# # Observations: 0 = No Umbrella, 1 = Umbrella
# observations = np.array([[0], [0], [1], [1], [0], [1], [0], [1]])
# # States: 0 = Sunny, 1 = Rainy
# # The hidden state sequence is not provided, we will train the HMM model to predict it
# # We will only use the observation sequence for training.

# # Step 2: Define and Train the HMM Model
# # Define the HMM: 2 hidden states (Sunny, Rainy), 2 possible observations (No Umbrella, Umbrella)
# model = hmm.MultinomialHMM(n_components=2, n_iter=1000)
# # Train the model on the observations (without the hidden state labels)
# model.fit(observations)

# # Step 3: Predict the hidden states (weather) given the observations
# predicted_states = model.predict(observations)

# # Step 4: Output results
# print("Predicted Hidden States (Weather):")
# print(predicted_states)

# # Step 5: Output the model's parameters (transition matrix, emission matrix)
# print("\nTransition Matrix (State to State):")
# print(model.transmat_)
# print("\nEmission Matrix (Observation | State):")
# print(model.emissionprob_)

# # Step 6: Making predictions for new observations (example)
# # Let's predict the weather based on a new sequence of umbrella usage
# new_observations = np.array([[0], [1], [1]])  # No Umbrella, Umbrella, Umbrella
# predicted_new_states = model.predict(new_observations)
# print("\nPredicted Hidden States for New Observations (No Umbrella, Umbrella, Umbrella):")
# print(predicted_new_states)

# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from hmmlearn import hmm

# Step 1: Load the real-world dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
data = pd.read_csv(url, header=None)

# Step 2: Preprocess the dataset
# Map the target labels (last column) to integers
label_mapping = {"R": 0, "M": 1}  # Rock (R) or Mine (M)
data.iloc[:, -1] = data.iloc[:, -1].map(label_mapping)

# Extract features and convert to required format
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels (hidden states)

# Convert continuous data to discrete states (required by MultinomialHMM)
# Discretize features using bins
n_bins = 10  # Number of bins for discretization
X_binned = np.digitize(X, bins=np.linspace(X.min(), X.max(), n_bins))

# Step 3: Train the HMM model
n_states = 2  # We assume two hidden states (Rock and Mine)
hmm_model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=1e-4, verbose=True)
hmm_model.fit(X_binned)

# Step 4: Make predictions using the trained model
log_prob, predicted_states = hmm_model.decode(X_binned, algorithm="viterbi")

# Step 5: Print results
print("Predicted States:", predicted_states)

# Step 6: Print trained model parameters
print("Trained Start Probabilities:\n", hmm_model.startprob_)
print("Trained Transition Matrix:\n", hmm_model.transmat_)
print("Trained Emission Probabilities:\n", hmm_model.emissionprob_)
