# Datamining Multi-Class and Multi-Label Classification,  k-means clustering algorithm and Class Imbalance in Binary Classification Project
## Project Overview
This project contains solutions to several problems related to classification and clustering:

Multi-Label Classification: Train and evaluate SVM models with different kernels (Gaussian and Polynomial) on the Scene dataset.
K-Means Clustering: Implement k-means clustering from scratch on the Seeds dataset.
Class Imbalance in Binary Classification: Analyze and train models on the German Credit Card dataset while addressing class imbalance.

## File Structure
Problem.py: Contains all scripts to handle multi-label classification with Gaussian and Polynomial kernels.
Problem2_FA24.ipynb: Notebook implementing k-means clustering.
Problem3_FA24.ipynb: Notebook solving binary classification for the German Credit Card dataset.
X_train.txt, X_test.txt, y_train.txt, y_test.txt: Datasets for multi-label classification.
seeds.txt: Datasets for K-mean Clustering
German Credit Data.txt, German Credit Data1 (1).txt, Dataset Documentation.txt: Datasets for Class Imbalance
README.md: Instructions for running the code.

## How to Run the Code
### Problem 1: Multi-Label Classification
Dependencies:

Python 3.x
NumPy
scikit-learn
Open Problem2_FA24.ipynb in a Jupyter Notebook or any compatible IDE (e.g., Google Colab).
Run all cells sequentially.

Expected Output: Displays accuracy percentages for both Gaussian and Polynomial kernels for all classes.
Outputs optimal hyperparameters for each kernel.

### Problem 2: K-Means Clustering
Dependencies:

Python 3.x
NumPy
Run the Notebook:

Open Problem2_FA24.ipynb in a Jupyter Notebook or any compatible IDE (e.g., Google Colab).
Run all cells sequentially.
Expected Output: Displays average Sum of Squared Errors (SSE) for k = 3, 5, 7 with 10 random initializations.

## Problem 3: Binary Classification with Class Imbalance
Dependencies:

Python 3.x
scikit-learn
NumPy
Pandas
Run the Notebook:

Open Problem3_FA24.ipynb in a Jupyter Notebook or any compatible IDE (e.g., Google Colab).
Run all cells sequentially.
Expected Output:

Displays the F-Score for the model on the test set.
Compares performance across techniques for handling class imbalance (e.g., SMOTE, re-sampling).

## Dataset Details
Scene Dataset:
Multi-label dataset with 6 classes.
Files: X_train.txt, X_test.txt, y_train.txt, y_test.txt.

Seeds Dataset:
Dataset of 210 samples with 7 attributes.
Files: seeds.txt

German Credit Card Dataset:
Binary classification dataset with 1,000 samples and 20 attributes.
Files: German Credit Data.txt, German Credit Data1 (1).txt, Dataset Documentation.txt

## Results Summary
Problem 1:
Gaussian Kernel Accuracy: 67.66%
Polynomial Kernel Accuracy: 61.26%
Problem 2:
Average SSE for k = 3: 587.9040
Average SSE for k = 5: 406.3259
Average SSE for k = 7: 307.0033
Problem 3:
F-Score: 0.82
Techniques for addressing class imbalance: SMOTE, cost-sensitive learning, etc.

## Troubleshooting
Missing Dependencies: Install missing libraries using pip:
pip install numpy scikit-learn pandas

Dataset Issues: Ensure the data files (X_train.txt, etc.) are placed in the same directory as the scripts.
