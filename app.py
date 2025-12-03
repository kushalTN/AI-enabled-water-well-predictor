# app.py

import streamlit as st
import pandas as pd
import os
import folium
from streamlit_folium import st_folium
from predict import make_predictions, PREDICTION_ASSETS

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="💧 AI Water Well Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💧 AI-Enabled Water Well Predictor")
st.markdown("An AI tool to predict **well suitability, aquifer depth, discharge, drilling method, and water quality**.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Home", "Live Predictor", "Bulk Prediction", "About"])

# --- Check if Models are Loaded ---
if PREDICTION_ASSETS is None:
    st.error("🚨 Models not found! Please run `python train_models.py` first.")
    st.stop()

# ==============================
# HOME PAGE
# ==============================
if choice == "Home":
    st.subheader("Project Overview")
    st.write("""
    This system integrates **NAQUIM-style data** with **Machine Learning models** to provide crucial predictions for water well construction.
    
    ### Key Predictions:
    - **Site Suitability**: Is the location viable for a well?
    - **Depth of Water-Bearing Zone**: How deep to drill for water?
    - **Expected Discharge**: What is the potential water yield (LPM)?
    - **Suitable Drilling Technique**: Which drilling method is best for the local geology?
    - **Expected Groundwater Quality**: Is the water likely to be good, moderate, or poor?

    Navigate to the **Live Predictor** to test a single location or to **Bulk Prediction** to process a whole file.
    """)
    st.image("https://i.imgur.com/m46TqKz.png", caption="System Workflow")

# ==============================
# LIVE PREDICTOR PAGE
# ==============================
elif choice == "Live Predictor":
    st.sidebar.header("Input Features")

    district = st.sidebar.selectbox("District", ['Prakasam', 'Guntur', 'West Godavari', 'East Godavari', 'Anantapur', 'Krishna', 'Chittoor', 'Kadapa', 'Kurnool', 'Nellore'])
    lithology = st.sidebar.selectbox("Lithology", ['Alluvium', 'Limestone', 'Granite', 'Sandstone', 'Basalt', 'Shale'])
    aquifer_type = st.sidebar.selectbox("Aquifer Type", ['Semi-confined', 'Unconfined', 'Fractured', 'Confined'])
    water_level = st.sidebar.slider("Water Level (m)", 0.0, 50.0, 10.0, help="Depth from ground level to water table.")
    ph = st.sidebar.slider("pH", 6.0, 9.0, 7.5)
    ec_us_cm = st.sidebar.slider("EC (uS/cm)", 200, 2000, 1000)
    hardness = st.sidebar.slider("Hardness (mg/L)", 100, 800, 300)
    nitrate = st.sidebar.slider("Nitrate (mg/L)", 10, 100, 45)
    fluoride = st.sidebar.slider("Fluoride (mg/L)", 0.0, 2.0, 1.0)

    st.subheader("🌍 Select Location on Map")
    
    if 'center' not in st.session_state:
        st.session_state.center = [15.9129, 79.7400]
    if 'location' not in st.session_state:
        st.session_state.location = None

    m = folium.Map(location=st.session_state.center, zoom_start=7)
    if st.session_state.location:
        folium.Marker(st.session_state.location, popup="Selected Location").add_to(m)
    
    map_data = st_folium(m, width=1000, height=500)

    # THIS 'IF' BLOCK IS THE FIX. It prevents the error by safely checking map data before use.
    if map_data and map_data.get("last_clicked"):
        st.session_state.location = [map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]]
    
    if st.session_state.location:
        st.success(f"Selected Location: Latitude={st.session_state.location[0]:.4f}, Longitude={st.session_state.location[1]:.4f}")
    else:
        st.info("Click a location on the map to select its coordinates.")

    predict_button = st.button("Get Predictions for Selected Location", type="primary", use_container_width=True, disabled=not st.session_state.location)

    if predict_button:
        input_data = pd.DataFrame([{
            'District': district,
            'Latitude': st.session_state.location[0],
                'Longitude': st.session_state.location[1],

            'Lithology': lithology,
            'Aquifer_Type': aquifer_type,
            'Water_Level_m': water_level,
            'pH': ph,
            'EC_uS_cm': ec_us_cm,
            'Hardness_mg_L': hardness,
            'Nitrate_mg_L': nitrate,
            'Fluoride_mg_L': fluoride,
        }])
        
        with st.spinner("🧠 AI is analyzing the data..."):
            predictions = make_predictions(input_data)
        
        st.markdown("### 🔎 Prediction Results")
        st.write("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Well Suitability", predictions['Suitability'])
            st.metric("Expected Depth (m)", predictions['Expected Depth (m)'])
            st.metric("Expected Discharge (LPM)", predictions['Expected Discharge (LPM)'])
        with col2:
            st.metric("Recommended Drilling Method", predictions['Recommended Drilling'])
            st.metric("Expected Water Quality", predictions['Expected Water Quality'])
        
        st.write("---")
        with st.expander("📝 Provide Feedback on These Predictions"):
            with st.form("feedback_form"):
                rating = st.slider("How accurate do you think these predictions are?", 1, 5, 3)
                feedback_text = st.text_area("Additional Feedback (Optional)")
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    st.success("✅ Thank you! Your feedback has been submitted.")

# ==============================
# BULK PREDICTION PAGE
# ==============================
elif choice == "Bulk Prediction":
    st.subheader("Upload a CSV File for Bulk Predictions")
    uploaded_file = st.file_uploader("Your file must contain the same columns as the training data.", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded Data Preview", df.head())

        if st.button("Run Bulk Prediction", type="primary"):
            with st.spinner("Processing all rows... This may take a moment."):
                results = []
                prediction_keys = [
                    'Suitability', 'Expected Depth (m)', 'Expected Discharge (LPM)',
                    'Recommended Drilling', 'Expected Water Quality'
                ]
                for i in range(len(df)):
                    row = df.iloc[[i]]
                    try:
                        predictions = make_predictions(row)
                        results.append(predictions)
                    except Exception as e:
                        error_dict = {key: "Error processing row" for key in prediction_keys}
                        results.append(error_dict)
                
                results_df = pd.DataFrame(results)
            
            st.write("🔮 Bulk Prediction Results")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='bulk_predictions.csv',
                mime='text/csv',
            )

# ==============================
# ABOUT PAGE
# ==============================
else:
    st.subheader("About This Project")
    st.write("""
    This application is an AI-enabled decision support system for water well construction, driven by NAQUIM data from the Central Ground Water Board (CGWB).
    
    - **Frontend**: Built with **Streamlit** for a user-friendly graphical interface.
    - **Backend**: Powered by **Python** and **scikit-learn**.
    - **Machine Learning Model**: Utilizes a **Random Forest** algorithm, a powerful ensemble method that combines multiple decision trees to improve prediction accuracy and control over-fitting.
    
    The system is designed to provide essential predictions to assist users in making informed decisions about drilling for groundwater.
    """)