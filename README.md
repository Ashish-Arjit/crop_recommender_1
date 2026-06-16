# 🌱 Precision Agriculture: Crop Recommendation System

A modular, beginner-friendly Crop Recommendation System built using Python, Scikit-learn, Pandas, NumPy, Matplotlib, and Streamlit. This project recommends the best-suited crop for cultivation based on soil nutrient measurements (Nitrogen, Phosphorus, Potassium, pH) and climatic indicators (Temperature, Humidity, Rainfall).

---

## 📂 Project Structure

```text
Crop_Recommendation_System/
│
├── dataset/
│   ├── Crop_recommendation.csv   # The crop dataset (2,200 samples)
│   └── plots/                    # Generated charts saved by the visualization pipeline
│       ├── crop_count_distribution.png
│       ├── correlation_heatmap.png
│       ├── temperature_distribution.png
│       └── rainfall_distribution.png
│
├── models/
│   └── crop_model.pkl            # Trained model artifact saved by joblib
│
├── src/
│   ├── import_libraries.py       # File 1: Import explanations & checks
│   ├── load_dataset.py           # File 2: Basic loading & exploratory statistics
│   ├── data_preprocessing.py     # File 3: Handling duplicates, nulls, and target separation
│   ├── data_visualization.py     # File 4: Matplotlib-based visualizations generator
│   ├── train_test_split_file.py  # File 5: Splits dataset into 80% train and 20% test sets
│   ├── train_model.py            # File 6: Trains Decision Tree vs Random Forest
│   ├── hyperparameter_tuning.py  # File 7: GridSearchCV tuning for Random Forest
│   ├── evaluate_model.py         # File 8: Accuracy, confusion matrix, classification report
│   ├── save_model.py             # File 9: Model serialization using joblib
│   └── predict_crop.py           # File 10: Standalone prediction on sample values
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies list
├── README.md                     # Setup instructions and deployment guide (This file)
└── report_content.md             # Complete report contents for viva / presentation
```

---

## ⚙️ Setup & Running Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Open your command terminal (Command Prompt, PowerShell, or Bash) in the project root directory and run:
```bash
pip install -r requirements.txt
```

### 3. Running the AI Pipeline Step-by-Step
Every file in `src/` can be run independently to show individual stages of the machine learning pipeline:
```bash
# File 1: Check libraries
python src/import_libraries.py

# File 2: Load and inspect raw dataset
python src/load_dataset.py

# File 3: Run data preprocessing & check missing/duplicate rows
python src/data_preprocessing.py

# File 4: Generate Matplotlib-based visualizations
python src/data_visualization.py

# File 5: Perform train-test split
python src/train_test_split_file.py

# File 6: Train base Decision Tree & Random Forest models
python src/train_model.py

# File 7: Run GridSearchCV to optimize Random Forest
python src/hyperparameter_tuning.py

# File 8: Evaluate model (prints classification report & confusion matrix)
python src/evaluate_model.py

# File 9: Train and save the best model to 'models/crop_model.pkl'
python src/save_model.py

# File 10: Test prediction on custom values
python src/predict_crop.py
```

### 4. Running the Streamlit Web Application
To run the interactive UI dashboard:
```bash
streamlit run app.py
```
This will open the web page automatically in your browser (typically at `http://localhost:8501`).

---

## ☁️ Streamlit Community Cloud Deployment Guide

Follow these steps to deploy this application online, making it publicly accessible to anyone:

### Step 1: Create a GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click the **New** button to create a new repository.
3. Name your repository (e.g., `crop-recommendation-system`).
4. Set visibility to **Public**.
5. Do NOT check "Add a README file" (we already have one).
6. Click **Create repository**.

### Step 2: Upload Project Files to GitHub
Open your local terminal inside the project directory and run:
```bash
# Initialize git
git init

# Add all files to staging area
git add .

# Create the initial commit
git commit -m "Initial commit of Crop Recommendation System"

# Rename branch to main
git branch -M main

# Link to your GitHub repository (replace with your repository's URL)
git remote add origin https://github.com/YOUR_USERNAME/crop-recommendation-system.git

# Push files to GitHub
git push -u origin main
```

### Step 3: Deploy to Streamlit Community Cloud
1. Visit [Streamlit Community Cloud](https://share.streamlit.io/) and click **Sign in** (use your GitHub account to sign in).
2. Click the **New app** button (or **Create app**).
3. Fill in the deployment details:
   - **Repository:** Choose `YOUR_USERNAME/crop-recommendation-system` (or search for it).
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**
5. Streamlit will set up the virtual environment, install the packages listed in `requirements.txt`, and load the app. This takes about 1-2 minutes.
6. Once deployed, your app will be publicly available at a link like:
   `https://crop-recommendation-system.streamlit.app`

### Step 4: Making Future Updates
If you update any code locally and want the website to reflect those changes:
1. Commit the changes and push them to GitHub:
   ```bash
   git add .
   git commit -m "Updated prediction styles"
   git push origin main
   ```
2. Streamlit Community Cloud detects the push to the `main` branch automatically and rebuilds/re-deploys the app within seconds.
