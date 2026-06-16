# ==========================================
# File 9: save_model.py
# Purpose: Save the trained and optimized model to a file using joblib.
# ==========================================

import os
import joblib
# Import evaluate_model from File 8 to get the best model
from evaluate_model import evaluate_model

def save_trained_model():
    """
    Function to save the trained model into a file.
    - Gets the best model using the evaluate_model pipeline.
    - Resolves the models/ output path dynamically.
    - Saves the model as 'models/crop_model.pkl'.
    - Prints a confirmation message.
    """
    # 1. Run evaluation to get the best trained Random Forest model
    best_rf, _ = evaluate_model()
    
    print("\n--- Saving the Model ---")
    
    # 2. Determine path to save the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "..", "models")
    
    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "crop_model.pkl")
    
    # 3. Save the model using joblib.dump()
    # joblib.dump serializes the Python object (model structure + trained weights)
    # and writes it to disk.
    joblib.dump(best_rf, model_save_path)
    
    print(f"\nSuccess: Trained model saved successfully!")
    print(f"Model saved at: {os.path.abspath(model_save_path)}")

if __name__ == "__main__":
    save_trained_model()
