# ==========================================
# File 6: train_model.py
# Purpose: Train ML models (Decision Tree and Random Forest) and display accuracies.
# ==========================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Import split_dataset from File 5 (train_test_split_file)
from train_test_split_file import split_dataset

def train_and_compare():
    """
    Function to train two different machine learning classifiers:
    1. Decision Tree Classifier
    2. Random Forest Classifier
    
    It evaluates both models and prints their training/testing accuracies.
    """
    # 1. Load split dataset
    X_train, X_test, y_train, y_test = split_dataset()
    
    print("\n--- Model Training & Comparison ---")
    
    # ==============================================================
    # Model 1: Decision Tree Classifier
    # Purpose/Concept:
    # A Decision Tree is a flowchart-like tree structure where:
    # - Internal nodes represent tests on features (e.g., is temperature > 25°C?).
    # - Branches represent outcomes of the test.
    # - Leaf nodes represent final class labels (the recommended crop).
    # It is simple, easy to interpret, but prone to overfitting on noisy data.
    # ==============================================================
    print("\n[Training Model 1: Decision Tree Classifier]")
    # Initialize the classifier (using random_state for consistency)
    dt_model = DecisionTreeClassifier(random_state=42)
    # Train (fit) the model on the training data
    dt_model.fit(X_train, y_train)
    
    # Predict on training and testing sets
    dt_train_preds = dt_model.predict(X_train)
    dt_test_preds = dt_model.predict(X_test)
    
    # Calculate accuracies
    dt_train_acc = accuracy_score(y_train, dt_train_preds)
    dt_test_acc = accuracy_score(y_test, dt_test_preds)
    
    print(f"Decision Tree - Training Accuracy: {dt_train_acc * 100:.2f}%")
    print(f"Decision Tree - Testing Accuracy:  {dt_test_acc * 100:.2f}%")
    
    # ==============================================================
    # Model 2: Random Forest Classifier
    # Purpose/Concept:
    # Random Forest is an ensemble learning method that builds a "forest"
    # of multiple independent Decision Trees.
    # - It trains each tree on a random subset of data (bagging) and features.
    # - For prediction, it combines the individual votes of all trees and
    #   recommends the class with the majority votes.
    # This reduces overfitting significantly and delivers much better accuracy.
    # ==============================================================
    print("\n[Training Model 2: Random Forest Classifier]")
    # Initialize the classifier (using random_state for consistency)
    rf_model = RandomForestClassifier(random_state=42)
    # Train (fit) the model on the training data
    rf_model.fit(X_train, y_train)
    
    # Predict on training and testing sets
    rf_train_preds = rf_model.predict(X_train)
    rf_test_preds = rf_model.predict(X_test)
    
    # Calculate accuracies
    rf_train_acc = accuracy_score(y_train, rf_train_preds)
    rf_test_acc = accuracy_score(y_test, rf_test_preds)
    
    print(f"Random Forest - Training Accuracy: {rf_train_acc * 100:.2f}%")
    print(f"Random Forest - Testing Accuracy:  {rf_test_acc * 100:.2f}%")
    
    return dt_model, rf_model

if __name__ == "__main__":
    train_and_compare()
