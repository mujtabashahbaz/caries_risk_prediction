import streamlit as st
import random

def calculate_risk_score(age, sugar_intake, brushing_frequency, flossing_frequency, 
                         fluoride_exposure, past_caries, salivary_flow_rate, salivary_ph):
    # Simple risk calculation based on weighted factors
    risk_score = 0
    risk_score += sugar_intake * 0.2
    risk_score += (3 - brushing_frequency) * 0.15
    risk_score += (2 - flossing_frequency) * 0.1
    risk_score += (1 - fluoride_exposure) * 0.1
    risk_score += past_caries * 0.2
    risk_score += (2 - salivary_flow_rate) * 0.15
    risk_score += (7.5 - salivary_ph) * 0.1
    
    # Normalize score to be between 0 and 1
    return min(max(risk_score / 10, 0), 1)

def main():
    st.title("AIDentify's Caries Risk Prediction Model")
    
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
        risk_score = calculate_risk_score(
            age, sugar_intake, brushing_frequency, flossing_frequency,
            int(fluoride_exposure), past_caries, salivary_flow_rate, salivary_ph
        )
        
        # Add a small random factor to make predictions less deterministic
        risk_score = min(max(risk_score + random.uniform(-0.05, 0.05), 0), 1)
        
        st.write(f"Caries Risk Score: {risk_score:.2f}")
        
        if risk_score > 0.6:
            st.warning("High risk of caries. Recommend preventive measures.")
        elif risk_score > 0.3:
            st.info("Moderate risk of caries. Consider additional preventive measures.")
        else:
            st.success("Low risk of caries. Maintain good oral hygiene.")
        
        st.write("Note: This is a simplified model for demonstration purposes only. "
                 "Consult with a dental professional for accurate risk assessment.")

if __name__ == "__main__":
    main()