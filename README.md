# AI-enabled-water-well-predictor
The web-based system uses NAQUIM data and AI to guide users on water well construction. It predicts site suitability, depth of water-bearing zones, expected discharge, drilling methods, and groundwater quality. With a user-friendly interface and feedback options, it offers reliable, location-specific decision support for groundwater management.


An AI-powered decision-support system that predicts **well suitability**, **aquifer (WBZ) depth**, **expected discharge (LPM)**, **drilling method**, and **water quality** using machine learning and geospatial data. :contentReference[oaicite:0]{index=0}  

The tool provides a **map-based interface** where users can select a location and instantly get groundwater predictions.

---

## Problem Statement

In many rural and semi-urban regions, borewell locations are chosen based on guesswork or limited local knowledge. This often results in:

- Dry or low-yield wells  
- Wasted drilling cost and effort  
- Continued water insecurity  

This project uses **Machine Learning** to make groundwater exploration **data-driven, transparent, and more reliable**.

---

##  Core Features

-  **Map-based location input** (Lat / Lon selected on OpenStreetMap)
-  **Well Suitability**: Suitable / Not Suitable
-  **Water-Bearing Zone (WBZ) Depth** prediction (meters)
-  **Expected Discharge** prediction (LPM)
-  **Recommended Drilling Method** (e.g., DTH, Rotary, Percussion, Auger)
-  **Water Quality Category**: Good / Moderate / Poor
-  Feature importance and model evaluation during training
-  Feedback logging via `feedback.csv`

---

##  Tech Stack

- **Language**: Python  
- **ML Libraries**: Scikit-Learn, XGBoost  
- **Data Handling**: Pandas, NumPy  
- **Visualization / UI**: Streamlit, Folium/Leaflet (through Streamlit components)  
- **Others**: Joblib / Pickle for model saving  

---

##  Project Structure

Based on your current folder:

```bash
AI-ENABLED WATER WELL PREDICTOR/
├── __pycache__/
├── data/
│   └── AP_KAR_NAQUIM_style_water_well_dataset_3000.csv
├── models/                     # Saved trained models (.pkl / .joblib)
├── venv/                       # (Optional) Python virtual environment
├── app.py                      # Main Streamlit application (map UI + predictions)
├── check_path.py               # Utility script to verify paths / environment
├── District_Statewise_Well.csv # Supporting statistics for districts
├── feedback file/              # (If folder) related to feedback handling
├── feedback.csv                # User feedback log from UI
├── final_nhs-wq_pre_2023_compressed.pdf  # Reference water quality report
├── predict.py                  # Standalone prediction script (CLI / batch)
└── train_models.py             # Model training & saving pipeline

 Dataset (High-Level)

Main training file:
data/AP_KAR_NAQUIM_style_water_well_dataset_3000.csv
Typical features include:
Location: Latitude, Longitude, District
Geology: Lithology, Aquifer type
Groundwater: Static water level, WBZ depth, discharge
Water Quality: pH, EC, Hardness, Nitrate, Fluoride
Labels / Targets:
Well_Suitability
Expected_WBZ_Depth_m
Expected_Discharge_LPM
Recommended_Drilling
Water_Quality_Class

