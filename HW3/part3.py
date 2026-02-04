import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# 1. Load Dataset
df = sns.load_dataset("tips")

# 2. Preprocessing
# We need to distinguish between numeric and categorical columns
# Numeric: 'total_bill', 'tip', 'size'
# Categorical: 'sex', 'smoker', 'day'
# Target: 'time'

# Encode categorical features and target using LabelEncoder
le = LabelEncoder()
cat_cols = ['sex', 'smoker', 'day']
target_col = 'time'

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df[target_col] = le.fit_transform(df[target_col])

# Organize X to have numeric columns first, then categorical columns
# This makes indexing easier inside the distance function
num_cols = ['total_bill', 'tip', 'size']
feature_order = num_cols + cat_cols

X = df[feature_order].values
y = df[target_col].values

# We store the number of numeric features to know where to split the array later
num_feature_count = len(num_cols)

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Implement Mixed Distance and Custom KNN
class MixedFeatureKNN:
    def __init__(self, k=3, w=1.0, num_idx_limit=3):
        self.k = k
        self.w = w
        self.num_idx_limit = num_idx_limit # Index where numeric features end
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _mixed_distance(self, row1, row2):
        # 1. Euclidean Distance for Numeric Features (indices 0 to num_idx_limit)
        # Formula: sqrt(sum((x - y)^2))
        num_part1 = row1[:self.num_idx_limit]
        num_part2 = row2[:self.num_idx_limit]
        dist_euclidean = np.sqrt(np.sum((num_part1 - num_part2)**2))

        # 2. Hamming Distance for Categorical Features (indices num_idx_limit to end)
        # Formula: sum(1 if x != y else 0)
        cat_part1 = row1[self.num_idx_limit:]
        cat_part2 = row2[self.num_idx_limit:]
        dist_hamming = np.sum(cat_part1 != cat_part2)

        # 3. Combine with weight W
        return dist_euclidean + (self.w * dist_hamming)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Calculate mixed distance to all training points
            distances = [self._mixed_distance(x, x_train) for x_train in self.X_train]
            
            # Sort and get top k neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)

# 4. Evaluation Loop (Varying K and W)
weights = [0.5, 1.0, 2.0, 3.0]
k_values = range(1, 21)

# Dictionary to store results: results[weight] = [acc_k1, acc_k2, ...]
results = {}

print(f"Running evaluation on {len(X_test)} test samples...")
print("-" * 40)

for w in weights:
    accuracies = []
    print(f"Processing Weight w={w}...")
    for k in k_values:
        knn = MixedFeatureKNN(k=k, w=w, num_idx_limit=num_feature_count)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
    results[w] = accuracies

print("-" * 40)
print("Done.")

# 5. Plotting Accuracy vs. K for different weights
plt.figure(figsize=(10, 7))

# Plot lines for each weight
for w in weights:
    plt.plot(k_values, results[w], marker='o', label=f'w={w}')

plt.title('Tips dataset: Accuracy vs. k for different categorical weights')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
