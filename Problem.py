#Problem 1

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load data
X_train = np.loadtxt('X_train.txt')
X_test = np.loadtxt('X_test.txt')
y_train = np.loadtxt('y_train.txt')
y_test = np.loadtxt('y_test.txt')

# Transpose the data
X_test_transposed = X_test.T

# Save the transposed data back to a file
np.savetxt('X_test_transposed.txt', X_test_transposed, fmt='%.6f')

print("File transposed successfully and saved as 'X_test_transposed.txt'")

#tried with only one class
clf = SVC(kernel='rbf', gamma='scale', C=1)
clf.fit(X_train, y_train[:, 0])  # Train on the first class
preds = clf.predict(X_test_transposed)
accuracy = np.mean(preds == y_test[:, 0]) * 100
print(f"Accuracy for Class 0 with RBF kernel: {accuracy:.2f}%")

#Accuracy with Gaussian (RBF) kernel

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load data
X_train = np.loadtxt('X_train.txt')
X_test = np.loadtxt('X_test.txt')
y_train = np.loadtxt('y_train.txt')
y_test = np.loadtxt('y_test.txt')

# Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test_transposed)

# Grid Search for Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale']
}

# Train SVM with optimized hyperparameters for each class
rbf_classifiers = []
for i in range(y_train.shape[1]):
    grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train[:, i])  # Optimize for each class
    best_params = grid.best_params_
    print(f"Class {i}: Best parameters: {best_params}")
    
    # Train with the best parameters
    clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], class_weight='balanced')
    clf.fit(X_train, y_train[:, i])
    rbf_classifiers.append(clf)

# Predict on the test set
rbf_predictions = np.zeros_like(y_test)
for i, clf in enumerate(rbf_classifiers):
    rbf_predictions[:, i] = clf.predict(X_test)

# Compute accuracy for multi-label classification
accuracies = []
for i in range(len(y_test)):
    T = y_test[i]
    P = rbf_predictions[i]
    numerator = np.sum(np.logical_and(T, P))
    denominator = np.sum(np.logical_or(T, P))
    if denominator > 0:
        accuracies.append(numerator / denominator)

rbf_accuracy = np.mean(accuracies) * 100
print(f"Accuracy with Gaussian (RBF) kernel: {rbf_accuracy:.2f}%")

#Accuracy with Polynomial kernel

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load data
X_train = np.loadtxt('X_train.txt')
X_test = np.loadtxt('X_test.txt')
y_train = np.loadtxt('y_train.txt')
y_test = np.loadtxt('y_test.txt')

# Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test_transposed)

# Hyperparameter Grid for Polynomial Kernel
param_grid = {
    'C': [0.1, 1, 10, 100]  # Regularization parameter
}

# Train SVM with optimized hyperparameters for each class
poly_classifiers = []
for i in range(y_train.shape[1]):
    grid = GridSearchCV(SVC(kernel='poly', degree=2, class_weight='balanced'), param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train[:, i])  # Optimize for each class
    best_params = grid.best_params_
    print(f"Class {i}: Best parameters: {best_params}")
    
    # Train with the best parameters
    clf = SVC(kernel='poly', degree=2, C=best_params['C'], class_weight='balanced')
    clf.fit(X_train, y_train[:, i])
    poly_classifiers.append(clf)

# Predict on the test set
poly_predictions = np.zeros_like(y_test)
for i, clf in enumerate(poly_classifiers):
    poly_predictions[:, i] = clf.predict(X_test)

# Compute accuracy for multi-label classification
accuracies = []
for i in range(len(y_test)):
    T = y_test[i]
    P = poly_predictions[i]
    numerator = np.sum(np.logical_and(T, P))
    denominator = np.sum(np.logical_or(T, P))
    if denominator > 0:
        accuracies.append(numerator / denominator)

poly_accuracy = np.mean(accuracies) * 100
print(f"Accuracy with Polynomial kernel: {poly_accuracy:.2f}%")

