# predict.py

import pandas as pd
import pickle
import os

MODEL_DIR = "models"

# --- Load all assets once ---
def load_prediction_assets():
    """Loads all models and the single column list."""
    assets = {"models": {}}
    try:
        model_names = ["suitability", "depth", "discharge", "drilling", "quality"]
        for name in model_names:
            with open(os.path.join(MODEL_DIR, f"{name}_model.pkl"), "rb") as f:
                assets["models"][name] = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, "training_columns.pkl"), "rb") as f:
            assets["columns"] = pickle.load(f)
            
    except FileNotFoundError:
        return None
        
    return assets

PREDICTION_ASSETS = load_prediction_assets()


# --- Prediction Function ---
def make_predictions(input_df: pd.DataFrame):
    """
    Takes a single-row DataFrame, preprocesses it, and returns all five predictions.
    """
    if PREDICTION_ASSETS is None:
        raise RuntimeError("Prediction assets are not loaded. Please train models first.")
    
    categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    training_columns = list(PREDICTION_ASSETS['columns'])
    aligned_input = input_encoded.reindex(columns=training_columns, fill_value=0)

    models = PREDICTION_ASSETS['models']
    
    predictions = {
        'Suitability': models['suitability'].predict(aligned_input)[0],
        'Expected Depth (m)': f"{models['depth'].predict(aligned_input)[0]:.2f}",
        'Expected Discharge (LPM)': f"{models['discharge'].predict(aligned_input)[0]:.2f}",
        'Recommended Drilling': models['drilling'].predict(aligned_input)[0],
        'Expected Water Quality': models['quality'].predict(aligned_input)[0]
    }
    
    return predictions