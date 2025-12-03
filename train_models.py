# train_models.py (FINAL FIX for Class Imbalance)

import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

# --- Configuration ---
DATA_PATH = "data/AP_KAR_NAQUIM_style_water_well_dataset_3000.csv"
MODEL_DIR = "models"

# --- Main Training Function ---
def train():
    """
    Trains models with a fix for class imbalance to ensure dynamic predictions.
    """
    print("🚀 Starting training with BALANCED class weights...")
    try:
        df = pd.read_csv(DATA_PATH).dropna()
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not read the CSV file: {e}")
        return

    # --- Feature Engineering ---
    features = df.drop(columns=[
        'Location', 'Well_Suitability', 'Expected_WBZ_Depth_m',
        'Expected_Discharge_LPM', 'Recommended_Drilling', 'Expected_Water_Quality'
    ])
    
    required_new_cols = ['Annual_Rainfall_mm', 'Slope_degrees']
    if all(col in features.columns for col in required_new_cols):
        print("✅ Advanced features found! Creating new intelligent features.")
        features['Lat_Lon_Interaction'] = features['Latitude'] * features['Longitude']
        features['Rainfall_Slope'] = features['Annual_Rainfall_mm'] / (features['Slope_degrees'] + 1)
    else:
        print("⚠ WARNING: Advanced features not found. Proceeding with basic features.")

    categorical_features = features.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(features, columns=categorical_features, drop_first=True)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "training_columns.pkl"), "wb") as f:
        pickle.dump(X_encoded.columns.tolist(), f)

    # --- Define Targets ---
    targets = {
        "Suitability": df['Well_Suitability'],
        "Depth": df['Expected_WBZ_Depth_m'],
        "Discharge": df['Expected_Discharge_LPM'],
        "Drilling": df['Recommended_Drilling'],
        "Quality": df['Expected_Water_Quality']
    }

    # --- THE FIX is applied below with class_weight='balanced' ---

    # Model 1: Well Suitability
    print("\n--- Training Model 1: Well Suitability ---")
    y = targets['Suitability']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    # FIX: Added class_weight='balanced' to handle imbalanced data
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'
    ).fit(X_train, y_train)
    print(f"  - Suitability Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    with open(os.path.join(MODEL_DIR, "suitability_model.pkl"), "wb") as f: pickle.dump(model, f)

    # Model 2: Expected Depth (XGBoost)
    print("\n--- Training Model 2: Expected Depth (XGBoost) ---")
    y = targets['Depth']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, y_train)
    print(f"  - Depth Model R² Score: {r2_score(y_test, model.predict(X_test)):.2f}")
    with open(os.path.join(MODEL_DIR, "depth_model.pkl"), "wb") as f: pickle.dump(model, f)
    
    # Model 3: Expected Discharge (XGBoost)
    print("\n--- Training Model 3: Expected Discharge (XGBoost) ---")
    y = targets['Discharge']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, y_train)
    print(f"  - Discharge Model R² Score: {r2_score(y_test, model.predict(X_test)):.2f}")
    with open(os.path.join(MODEL_DIR, "discharge_model.pkl"), "wb") as f: pickle.dump(model, f)

    # Model 4: Recommended Drilling
    print("\n--- Training Model 4: Recommended Drilling ---")
    y = targets['Drilling']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    # FIX: Added class_weight='balanced' to handle imbalanced data
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'
    ).fit(X_train, y_train)
    print(f"  - Drilling Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    with open(os.path.join(MODEL_DIR, "drilling_model.pkl"), "wb") as f: pickle.dump(model, f)

    # Model 5: Expected Water Quality
    print("\n--- Training Model 5: Expected Water Quality ---")
    y = targets['Quality']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    # FIX: Added class_weight='balanced' to handle imbalanced data
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'
    ).fit(X_train, y_train)
    print(f"  - Quality Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    with open(os.path.join(MODEL_DIR, "quality_model.pkl"), "wb") as f: pickle.dump(model, f)
        
    print("\n🎉 All models trained with BALANCED weights and saved successfully!")

if __name__ == "__main__":
    train()