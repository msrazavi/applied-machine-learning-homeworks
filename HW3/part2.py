import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# 1. Load Dataset from UCI-
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, names=column_names)

# 2. Preprocessing (Label Encoding)
# KNN requires numerical input to calculate distances (even Hamming).
# We transform categorical strings ('vhigh', 'low', etc.) into integers.
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('class', axis=1).values
y = df['class'].values

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Implement Hamming Distance and Manual KNN
def hamming_distance(point1, point2):
    """
    Calculates the Hamming distance between two numpy arrays.
    Distance = Number of positions where symbols differ.
    """
    # Sum of elements where point1 is NOT equal to point2
    return np.sum(point1 != point2)

class ManualKNN_Hamming:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        # 1. Compute Hamming distances
        distances = [hamming_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Get k nearest indices
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Get labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Majority Vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# 4. Evaluation Loop (Manual vs Sklearn)
k_values = range(1, 21)
manual_accuracies = []
sklearn_accuracies = []

print(f"{'k':<5} | {'Manual Acc':<12} | {'Sklearn Acc':<12}")
print("-" * 35)

for k in k_values:
    # --- Manual KNN (Hamming) ---
    manual_knn = ManualKNN_Hamming(k=k)
    manual_knn.fit(X_train, y_train)
    manual_preds = manual_knn.predict(X_test)
    manual_acc = accuracy_score(y_test, manual_preds)
    manual_accuracies.append(manual_acc)

    # --- Sklearn KNN (Hamming) ---
    # metric='hamming' in sklearn usually calculates the proportion (0 to 1).
    # However, the ranking of neighbors is identical to the count approach.
    sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric='hamming')
    sklearn_knn.fit(X_train, y_train)
    sklearn_preds = sklearn_knn.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_preds)
    sklearn_accuracies.append(sklearn_acc)
    
    print(f"{k:<5} | {manual_acc:.4f}       | {sklearn_acc:.4f}")

# 5. Plotting Accuracy vs. K
plt.figure(figsize=(9, 6))

# Plot Manual KNN
plt.plot(k_values, manual_accuracies, label='Manual KNN', 
         marker='o', linestyle='--', color='#2b7bba')

# Plot Sklearn KNN
plt.plot(k_values, sklearn_accuracies, label='sklearn KNN', 
         marker='s', linestyle=':', color='#eb8c23')

plt.title('Car Evaluation dataset: Manual vs. sklearn KNN')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
