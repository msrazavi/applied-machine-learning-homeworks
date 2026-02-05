import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# 1. & 2. Load Dataset and Split
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data (80% train, 20% test)
# random_state=42 ensures we get the same split every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Implement Euclidean Distance and Manual KNN
def euclidean_distance(point1, point2):
    # Calculates the Euclidean distance between two numpy arrays.
    # Formula: sqrt(sum((x - y)^2))
    return np.sqrt(np.sum((point1 - point2)**2))

class ManualKNN:
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
        # 1. Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Return the most common class label (Majority Vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# 4. Evaluation Loop (Manual vs Sklearn)
k_values = range(1, 21)
manual_accuracies = []
sklearn_accuracies = []

print(f"{'k':<5} | {'Manual Acc':<12} | {'Sklearn Acc':<12}")
print("-" * 35)

for k in k_values:
    # --- Manual KNN ---
    manual_knn = ManualKNN(k=k)
    manual_knn.fit(X_train, y_train)
    manual_preds = manual_knn.predict(X_test)
    manual_acc = accuracy_score(y_test, manual_preds)
    manual_accuracies.append(manual_acc)

    # --- Sklearn KNN ---
    # metric='minkowski' with p=2 is equivalent to Euclidean distance
    sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    sklearn_knn.fit(X_train, y_train)
    sklearn_preds = sklearn_knn.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_preds)
    sklearn_accuracies.append(sklearn_acc)
    
    print(f"{k:<5} | {manual_acc:.4f}       | {sklearn_acc:.4f}")

# 5. Plotting Accuracy vs. K
plt.figure(figsize=(9, 6))

# Plot Manual KNN (Blue circles, larger)
plt.plot(k_values, manual_accuracies, label='Manual KNN', 
         marker='o', markersize=9, linestyle='--', color='blue')

# Plot Sklearn KNN (Orange squares, smaller - to show overlap/difference)
plt.plot(k_values, sklearn_accuracies, label='sklearn KNN', 
         marker='s', markersize=5, linestyle=':', color='orange')

plt.title('Breast Cancer dataset: Manual vs. sklearn KNN')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 22, 2.5)) # Adjusting x-ticks to look like the sample
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
