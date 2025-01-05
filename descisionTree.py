import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Common Utility Functions
# ----------------------------

# Split dataset based on an attribute and a threshold
def split_dataset(X, y, feature_index, threshold):
    left_indices = [i for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_indices = [i for i in range(len(X)) if X[i][feature_index] > threshold]
    return left_indices, right_indices

# Calculate F1 and F2 scores
def calculate_f1_f2(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate F2 score
    beta = 2
    if (beta**2 * precision + recall) == 0:
        f2 = 0.0
    else:
        f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    return f1, f2

# Predict a single instance
def predict_instance(node, instance):
    if node.value is not None:
        return node.value
    if instance[node.feature_index] <= node.threshold:
        return predict_instance(node.left, instance)
    else:
        return predict_instance(node.right, instance)

# Predict for a dataset
def predict(tree, X):
    return [predict_instance(tree, instance) for instance in X]

# Load and preprocess dataset
def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)

    # Separate features (X) and target labels (y)
    X = data.iloc[:, :-1].values  # All columns except the last one (features)
    y = data.iloc[:, -1].values   # The last column (target)

    # Map target labels to integers if they are categorical
    if isinstance(y[0], str):  # Only do this if the labels are strings (like in Iris)
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = np.array([label_mapping[label] for label in y])

    # Shuffle dataset
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Train-test split (80% train, 20% test)
    split_index = int(0.8 * len(y))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

# Normalize the dataset and calculate distance to heaven (d2h)
def calculate_d2h(X):
    scaler = MinMaxScaler()
    normalized_X = scaler.fit_transform(X)
    heaven = np.zeros(normalized_X.shape[1])

    def d2h(example):
        return np.sqrt(np.sum((heaven - example) ** 2)) / np.sqrt(len(heaven))

    return np.array([d2h(x) for x in normalized_X])

# K-Fold Validation
def k_fold_validation(X, y, impurity_function, k=5, max_depth=3):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies, f1_scores, f2_scores, d2h_scores = [], [], [], []

    for train_index, val_index in kf.split(X):
        # Split data into training and validation sets
        X_train, X_val = [X[i] for i in train_index], [X[i] for i in val_index]
        y_train, y_val = [y[i] for i in train_index], [y[i] for i in val_index]

        # Build the decision tree on the training set
        tree = build_tree(X_train, y_train, impurity_function, max_depth=max_depth)

        # Make predictions on the validation set
        predictions = predict(tree, X_val)

        # Calculate accuracy
        accuracy = sum(1 for true, pred in zip(y_val, predictions) if true == pred) / len(y_val)
        accuracies.append(accuracy)

        # Calculate F1 and F2 scores
        f1, f2 = calculate_f1_f2(y_val, predictions)
        f1_scores.append(f1)
        f2_scores.append(f2)

        # Calculate distance to heaven (d2h)
        d2h_val = calculate_d2h(X_val)
        d2h_scores.append(np.mean(d2h_val))

    # Return average accuracy, F1, F2, and d2h scores across all folds
    return np.mean(accuracies), np.mean(f1_scores), np.mean(f2_scores), np.mean(d2h_scores)

# ----------------------------
# Impurity Calculation Functions
# ----------------------------

# Calculate entropy
def calculate_entropy(y):
    class_counts = Counter(y)
    total_samples = len(y)
    entropy = 0.0
    for count in class_counts.values():
        probability = count / total_samples
        entropy -= probability * np.log2(probability)
    return entropy

# Calculate misclassification error
def calculate_misclassification_error(y):
    class_counts = Counter(y)
    total_samples = len(y)
    max_class_count = max(class_counts.values())
    return 1 - (max_class_count / total_samples)

# Calculate information gain using a given impurity function
def information_gain(X, y, feature_index, threshold, impurity_function):
    parent_impurity = impurity_function(y)
    left_indices, right_indices = split_dataset(X, y, feature_index, threshold)
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0
    n = len(y)
    n_left, n_right = len(left_indices), len(right_indices)
    left_impurity = impurity_function([y[i] for i in left_indices])
    right_impurity = impurity_function([y[i] for i in right_indices])
    weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
    return parent_impurity - weighted_impurity

# ----------------------------
# Decision Tree Functions
# ----------------------------

# Node class for the decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Find the best split using a given impurity function
def find_best_split(X, y, impurity_function):
    best_feature_index = None
    best_threshold = None
    best_gain = -1
    n_features = len(X[0])
    for feature_index in range(n_features):
        thresholds = set(row[feature_index] for row in X)
        for threshold in thresholds:
            gain = information_gain(X, y, feature_index, threshold, impurity_function)
            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold
    return best_feature_index, best_threshold, best_gain

# Build the decision tree using a given impurity function
def build_tree(X, y, impurity_function, depth=0, max_depth=10):
    if len(set(y)) == 1 or depth == max_depth:
        most_common_label = Counter(y).most_common(1)[0][0]
        return Node(value=most_common_label)
    feature_index, threshold, gain = find_best_split(X, y, impurity_function)
    if gain == 0:
        most_common_label = Counter(y).most_common(1)[0][0]
        return Node(value=most_common_label)
    left_indices, right_indices = split_dataset(X, y, feature_index, threshold)
    left_subtree = build_tree([X[i] for i in left_indices], [y[i] for i in left_indices], impurity_function, depth + 1, max_depth)
    right_subtree = build_tree([X[i] for i in right_indices], [y[i] for i in right_indices], impurity_function, depth + 1, max_depth)
    return Node(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)

# Print the decision tree
def print_tree(node, depth=0):
    if node.value is not None:
        print(f"{'|   ' * depth}Leaf: Class = {node.value}")
    else:
        print(f"{'|   ' * depth}Feature {node.feature_index} <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

# ----------------------------
# Main Function
# ----------------------------

if __name__ == "__main__":
    # Load and preprocess the dataset (Iris dataset)
    X_train, X_test, y_train, y_test = load_and_preprocess_data("iris.csv")  # Ensure "iris.csv" is in the working directory

    # Perform K-Fold validation using entropy
    print("Evaluating Decision Tree using Entropy as Impurity Measure...\n")
    avg_accuracy_entropy, avg_f1_entropy, avg_f2_entropy, avg_d2h_entropy = k_fold_validation(X_train.tolist(), y_train.tolist(), calculate_entropy, k=5, max_depth=3)
    print(f"Average Accuracy with 5-Fold Cross-Validation (Entropy): {avg_accuracy_entropy * 100:.2f}%")
    print(f"Average F1 Score with 5-Fold Cross-Validation (Entropy): {avg_f1_entropy * 100:.2f}%")
    print(f"Average F2 Score with 5-Fold Cross-Validation (Entropy): {avg_f2_entropy * 100:.2f}%")
    print(f"Average Distance to Heaven (d2h) with 5-Fold Cross-Validation (Entropy): {avg_d2h_entropy * 100:.2f}%\n")

    # Perform K-Fold validation using misclassification error
    print("Evaluating Decision Tree using Misclassification Error as Impurity Measure...\n")
    avg_accuracy_misclassification, avg_f1_misclassification, avg_f2_misclassification, avg_d2h_misclassification = k_fold_validation(
        X_train.tolist(), y_train.tolist(), calculate_misclassification_error, k=5, max_depth=3
    )
    print(f"Average Accuracy with 5-Fold Cross-Validation (Misclassification Error): {avg_accuracy_misclassification * 100:.2f}%")
    print(f"Average F1 Score with 5-Fold Cross-Validation (Misclassification Error): {avg_f1_misclassification * 100:.2f}%")
    print(f"Average F2 Score with 5-Fold Cross-Validation (Misclassification Error): {avg_f2_misclassification * 100:.2f}%")
    print(f"Average Distance to Heaven (d2h) with 5-Fold Cross-Validation (Misclassification Error): {avg_d2h_misclassification * 100:.2f}%\n")

    # Build a tree and print its structure
    print("Decision Tree Structure (Entropy):\n")
    decision_tree = build_tree(X_train.tolist(), y_train.tolist(), calculate_entropy, max_depth=3)
    print_tree(decision_tree)
