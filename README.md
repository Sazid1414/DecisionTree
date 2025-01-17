The code defines a decision tree implementation, evaluates it using K-Fold cross-validation with entropy and misclassification error as impurity measures, and prints the tree structure. Below is a line-by-line explanation, function by function:

---

### **Utility Functions**
#### **1. `split_dataset`**
Splits a dataset into two subsets based on a feature index and a threshold.

```python
def split_dataset(X, y, feature_index, threshold):
    left_indices = [i for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_indices = [i for i in range(len(X)) if X[i][feature_index] > threshold]
    return left_indices, right_indices
```

- **`left_indices`**: Finds indices of samples where the value of `feature_index` is less than or equal to `threshold`.
- **`right_indices`**: Finds indices where the value of `feature_index` is greater than `threshold`.
- **Return**: Lists of indices for the left and right splits.

---

#### **2. `calculate_f1_f2`**
Calculates F1 and F2 scores for a prediction.

```python
def calculate_f1_f2(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    beta = 2
    if (beta**2 * precision + recall) == 0:
        f2 = 0.0
    else:
        f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f1, f2
```

- **`precision_score`**: Measures the proportion of true positives among predicted positives.
- **`recall_score`**: Measures the proportion of true positives among actual positives.
- **`f1_score`**: Harmonic mean of precision and recall.
- **F2-Score**: Weighted by β² to emphasize recall more heavily.

---

#### **3. `predict_instance`**
Predicts the class for a single sample by traversing the decision tree.

```python
def predict_instance(node, instance):
    if node.value is not None:
        return node.value
    if instance[node.feature_index] <= node.threshold:
        return predict_instance(node.left, instance)
    else:
        return predict_instance(node.right)
```

- **Leaf node**: Returns the `value` (predicted class) if the node is a leaf.
- **Internal node**: Recursively traverses to the left or right subtree based on the `threshold`.

---

#### **4. `predict`**
Predicts the class for multiple samples by calling `predict_instance`.

```python
def predict(tree, X):
    return [predict_instance(tree, instance) for instance in X]
```

- Applies `predict_instance` to each instance in `X`.

---

#### **5. `load_and_preprocess_data`**
Loads a dataset from a file and preprocesses it.

```python
def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    if isinstance(y[0], str):
        label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
        y = np.array([label_mapping[label] for label in y])
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    split_index = int(0.8 * len(y))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test
```

- **`pd.read_csv`**: Reads the dataset.
- **Features and labels**: Separates input features (`X`) and target labels (`y`).
- **Categorical labels**: Converts string labels to integers using a mapping.
- **Shuffle**: Randomizes the order of the dataset.
- **Train-test split**: Splits into training (80%) and testing (20%) sets.

---

#### **6. `k_fold_validation`**
Performs K-Fold cross-validation.

```python
def k_fold_validation(X, y, impurity_function, k=5, max_depth=3):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies, f1_scores, f2_scores = [], [], []
    for train_index, val_index in kf.split(X):
        X_train, X_val = [X[i] for i in train_index], [X[i] for i in val_index]
        y_train, y_val = [y[i] for i in train_index], [y[i] for i in val_index]
        tree = build_tree(X_train, y_train, impurity_function, max_depth=max_depth)
        predictions = predict(tree, X_val)
        accuracy = sum(1 for true, pred in zip(y_val, predictions) if true == pred) / len(y_val)
        accuracies.append(accuracy)
        f1, f2 = calculate_f1_f2(y_val, predictions)
        f1_scores.append(f1)
        f2_scores.append(f2)
    return np.mean(accuracies), np.mean(f1_scores), np.mean(f2_scores)
```

- **`KFold`**: Splits the dataset into `k` folds.
- **Train-validation split**: Separates training and validation data.
- **Build tree**: Trains the tree on the training set.
- **Evaluate**: Computes accuracy, F1, and F2 scores for predictions on the validation set.

---

### **Impurity Functions**
#### **1. `calculate_entropy`**
Computes entropy for a set of labels.

```python
def calculate_entropy(y):
    class_counts = Counter(y)
    total_samples = len(y)
    entropy = 0.0
    for count in class_counts.values():
        probability = count / total_samples
        entropy -= probability * np.log2(probability)
    return entropy
```

---

#### **2. `calculate_misclassification_error`**
Computes the misclassification error.

```python
def calculate_misclassification_error(y):
    class_counts = Counter(y)
    total_samples = len(y)
    max_class_count = max(class_counts.values())
    return 1 - (max_class_count / total_samples)
```

---

### **Decision Tree Functions**
#### **1. `Node`**
Defines a tree node.

```python
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
```

---

#### **2. `find_best_split`**
Finds the optimal feature and threshold for splitting.

```python
def find_best_split(X, y, impurity_function):
    best_feature_index, best_threshold, best_gain = None, None, -1
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
```

---

#### **3. `build_tree`**
Recursively builds a decision tree.

```python
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
```

---

#### **4. `print_tree`**
Prints the structure of the decision tree.

```python
def print_tree(node, depth=0):
    if node.value is not None:
        print(f"{'|   ' * depth}Leaf: Class = {node.value}")
    else:
        print(f"{'|   ' * depth}Feature {node.feature_index} <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)
```

---

### **Main Execution**
1. **Load data**: `load_and_preprocess_data("heart.csv")`.
2. **K-Fold validation**:
   - With entropy: `k_fold_validation(calculate_entropy)`.
   - With misclassification error: `k_fold_validation(calculate_misclassification_error)`.
3. **Tree visualization**: Prints the decision tree.

