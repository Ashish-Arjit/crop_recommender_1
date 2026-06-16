# ==========================================
# File 3: data_preprocessing.py
# Purpose: Preprocess the dataset (handling nulls, duplicates, and split features/target).
# ==========================================

import pandas as pd
# Import the load_data function from File 2 (load_dataset)
from load_dataset import load_data

def preprocess_data():
    """
    Function to preprocess the Crop Recommendation dataset.
    - Loads the dataset.
    - Checks for missing values (nulls).
    - Checks and removes duplicate rows.
    - Splits the dataset into Features (X) and Target (y).
    """
    # Load dataset using our helper function
    df = load_data()
    
    print("\n--- Starting Data Preprocessing ---")
    
    # 1. Checking for missing (null) values in each column
    print("\nChecking for missing values in the dataset:")
    null_counts = df.isnull().sum()
    print(null_counts)
    
    # Check if there are any null values overall
    if null_counts.sum() > 0:
        print("Warning: Missing values found. Handling them by dropping rows with nulls...")
        df = df.dropna()
    else:
        print("Excellent: No missing values found in the dataset.")
        
    # 2. Checking for duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows found: {duplicate_count}")
    
    # If duplicates are found, remove them
    if duplicate_count > 0:
        print("Removing duplicates...")
        df = df.drop_duplicates()
        print(f"Dataset shape after removing duplicates: {df.shape}")
    else:
        print("No duplicate rows to remove.")
        
    # 3. Separating features (independent variables) and target (dependent variable)
    # Features (X): soil nutrients and weather metrics
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    
    # Target (y): the crop label to recommend
    y = df['label']
    
    print("\nPreprocessed successfully:")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    return X, y

if __name__ == "__main__":
    X, y = preprocess_data()
    print("\nSample Features (First 5 rows):")
    print(X.head())
    print("\nSample Target (First 5 values):")
    print(y.head())
