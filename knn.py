import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter


# Euclidean Distance Function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Classifier
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = self._predict(x)
            y_pred.append(prediction)
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distance = euclidean_distance(x, x_train)
            distances.append(distance)

        sorted_indices = np.argsort(distances)
        k_indices = []
        for i in range(self.k):
            k_indices.append(sorted_indices[i])

        k_labels = []
        for i in k_indices:
            k_labels.append(self.y_train[i])

        label_counts = Counter(k_labels)
        most_common_label = label_counts.most_common(1)[0][0]
        return most_common_label

# KNN Regressor
class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = self._predict(x)
            y_pred.append(prediction)
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distance = euclidean_distance(x, x_train)
            distances.append(distance)

        sorted_indices = np.argsort(distances)
        k_indices = []
        for i in range(self.k):
            k_indices.append(sorted_indices[i])

        k_targets = []
        for i in k_indices:
            k_targets.append(self.y_train[i])

        average_value = sum(k_targets) / len(k_targets)
        return average_value

# -------------------------
# Main Program
# -------------------------
if __name__ == "__main__":

    # ----------- Classification Test ------------
    iris = datasets.load_iris()
    X_class, y_class = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    clf = KNNClassifier(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            correct += 1
    accuracy = correct / len(predictions)
    print("KNN Classification Accuracy:", accuracy)

    # ----------- Regression Test ------------
    X_reg, y_reg = datasets.make_regression(n_samples=100, n_features=1, noise=15, random_state=1)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg = KNNRegressor(k=3)
    reg.fit(X_train_r, y_train_r)
    predictions_r = reg.predict(X_test_r)

    mse = 0
    for i in range(len(y_test_r)):
        mse += (y_test_r[i] - predictions_r[i]) ** 2
    mse /= len(y_test_r)
    print("KNN Regression MSE:", mse)