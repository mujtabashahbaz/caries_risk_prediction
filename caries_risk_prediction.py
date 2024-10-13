import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Generate synthetic data
def generate_synthetic_data(n_samples=10000):
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(5, 80, n_samples),
        'sugar_intake': np.random.randint(0, 10, n_samples),
        'brushing_frequency': np.random.randint(0, 3, n_samples),
        'flossing_frequency': np.random.randint(0, 2, n_samples),
        'fluoride_exposure': np.random.randint(0, 2, n_samples),
        'past_caries': np.random.randint(0, 5, n_samples),
        'salivary_flow_rate': np.random.uniform(0.1, 2.0, n_samples),
        'salivary_ph': np.random.uniform(5.5, 7.5, n_samples)
    })
    
    # Generate target variable (caries risk: 0 - low, 1 - moderate, 2 - high)
    data['caries_risk'] = np.where(
        (data['brushing_frequency'] == 0) | (data['sugar_intake'] >= 8) | (data['past_caries'] >= 4), 2,
        np.where(
            (data['brushing_frequency'] <= 1) | (data['sugar_intake'] >= 6) | (data['past_caries'] >= 2) |
            (data['salivary_flow_rate'] < 0.7) | (data['salivary_ph'] < 6.0), 1, 0
        )
    )
    
    return data

# Train the model
def train_model(data):
    X = data.drop('caries_risk', axis=1)
    y = data['caries_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Streamlit app
def main():
    st.title("Improved Caries Risk Prediction Model")
    
    # Check if model exists, if not, train and save it
    if not os.path.exists('caries_model.joblib') or not os.path.exists('scaler.joblib'):
        data = generate_synthetic_data()
        model, scaler = train_model(data)
        joblib.dump(model, 'caries_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
    else:
        model = joblib.load('caries_model.joblib')
        scaler = joblib.load('scaler.joblib')
    
    st.write("Enter patient information:")
    
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    sugar_intake = st.slider("Sugar Intake (0-10)", 0, 10, 5)
    brushing_frequency = st.selectbox("Brushing Frequency (times per day)", [0, 1, 2, 3])
    flossing_frequency = st.selectbox("Flossing Frequency (times per day)", [0, 1, 2])
    fluoride_exposure = st.checkbox("Regular Fluoride Exposure")
    past_caries = st.number_input("Number of Past Caries", min_value=0, max_value=10, value=0)
    salivary_flow_rate = st.slider("Salivary Flow Rate (mL/min)", 0.1, 2.0, 1.0, 0.1)
    salivary_ph = st.slider("Salivary pH", 5.5, 7.5, 6.5, 0.1)
    
    if st.button("Predict Caries Risk"):
        input_data = np.array([[age, sugar_intake, brushing_frequency, flossing_frequency,
                                int(fluoride_exposure), past_caries, salivary_flow_rate, salivary_ph]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict_proba(input_data_scaled)[0]
        
        st.write(f"Probability of Low Caries Risk: {prediction[0]:.2f}")
        st.write(f"Probability of Moderate Caries Risk: {prediction[1]:.2f}")
        st.write(f"Probability of High Caries Risk: {prediction[2]:.2f}")
        
        risk_level = np.argmax(prediction)
        if risk_level == 0:
            st.success("Low risk of caries. Maintain good oral hygiene.")
        elif risk_level == 1:
            st.warning("Moderate risk of caries. Consider additional preventive measures.")
        else:
            st.error("High risk of caries. Immediate preventive measures recommended.")
        
        st.write("Note: This model is for demonstration purposes. Always consult with a dental professional for accurate risk assessment.")

if __name__ == "__main__":
    main()