import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# [Previous functions remain unchanged]

def explain_results(risk_level, prediction, input_data):
    st.subheader("Explanation of Results")
    
    risk_levels = ["Low", "Moderate", "High"]
    st.write(f"Your caries risk level is: **{risk_levels[risk_level]}**")
    
    st.write(f"Probability breakdown:")
    for i, level in enumerate(risk_levels):
        st.write(f"- {level} risk: {prediction[i]:.2f}")
    
    st.write("\nKey factors influencing your risk level:")
    
    if input_data[0][2] < 2:  # Brushing frequency
        st.write("- Your brushing frequency is lower than recommended. Aim for brushing at least twice a day.")
    
    if input_data[0][1] > 5:  # Sugar intake
        st.write("- Your sugar intake is relatively high. Consider reducing sugary foods and drinks.")
    
    if input_data[0][3] == 0:  # Flossing frequency
        st.write("- You're not flossing regularly. Daily flossing can significantly reduce your caries risk.")
    
    if not input_data[0][4]:  # Fluoride exposure
        st.write("- You're not getting regular fluoride exposure. Consider using fluoride toothpaste or other fluoride treatments.")
    
    if input_data[0][5] > 2:  # Past caries
        st.write("- You have a history of multiple caries. This increases your risk for future caries.")
    
    if input_data[0][6] < 0.7:  # Salivary flow rate
        st.write("- Your salivary flow rate is lower than average. This can increase your caries risk.")
    
    if input_data[0][7] < 6.5:  # Salivary pH
        st.write("- Your salivary pH is on the acidic side. This can increase your risk of tooth demineralization.")
    
    st.write("\nRecommendations:")
    if risk_level == 0:
        st.write("- Maintain your good oral hygiene habits.")
        st.write("- Continue regular dental check-ups.")
    elif risk_level == 1:
        st.write("- Improve your brushing and flossing routine.")
        st.write("- Consider using fluoride mouthwash.")
        st.write("- Reduce sugar intake if it's high.")
        st.write("- Schedule a dental check-up in the near future.")
    else:
        st.write("- Significantly improve your oral hygiene routine.")
        st.write("- Use a fluoride toothpaste and consider additional fluoride treatments.")
        st.write("- Drastically reduce sugar intake.")
        st.write("- Schedule a dental appointment as soon as possible.")
    
    st.write("\nRemember: This is a simplified model for educational purposes. Always consult with a dental professional for personalized advice and treatment.")

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
        
        risk_level = np.argmax(prediction)
        if risk_level == 0:
            st.success("Low risk of caries.")
        elif risk_level == 1:
            st.warning("Moderate risk of caries.")
        else:
            st.error("High risk of caries.")
        
        explain_results(risk_level, prediction, input_data)

if __name__ == "__main__":
    main()