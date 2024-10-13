import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Function to generate synthetic data (update this with 10 features)
def generate_synthetic_data():
    np.random.seed(42)
    data_size = 1000
    
    age = np.random.randint(1, 100, data_size)
    sugary_diet = np.random.randint(0, 11, data_size)
    brushing_frequency = np.random.randint(0, 4, data_size)
    flossing_frequency = np.random.randint(0, 3, data_size)
    fluoride_exposure = np.random.choice([0, 1], data_size)
    dry_mouth = np.random.choice([0, 1], data_size)
    snacking_frequency = np.random.randint(0, 11, data_size)
    dental_visits = np.random.randint(0, 5, data_size)
    nighttime_bottle_feeding = np.random.choice([0, 1], data_size)
    orthodontic_appliances = np.random.choice([0, 1], data_size)
    
    labels = np.random.choice([0, 1, 2], data_size)  # 0 = low risk, 1 = moderate risk, 2 = high risk
    
    features = np.vstack((age, sugary_diet, brushing_frequency, flossing_frequency, fluoride_exposure, dry_mouth, 
                          snacking_frequency, dental_visits, nighttime_bottle_feeding, orthodontic_appliances)).T
    
    return pd.DataFrame(features, columns=["age", "sugary_diet", "brushing_frequency", "flossing_frequency",
                                           "fluoride_exposure", "dry_mouth", "snacking_frequency", 
                                           "dental_visits", "nighttime_bottle_feeding", "orthodontic_appliances"]), labels

# Function to train the model and scaler
def train_model(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

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
    
    if input_data[0][3] == 0:  # Flossing frequency
        st.write("- You're not flossing regularly. Daily flossing can significantly reduce your caries risk.")
    
    if input_data[0][1] > 5:  # Sugary diet
        st.write("- Your diet is high in sugary foods and beverages. Consider reducing sugary foods and drinks.")
    
    if not input_data[0][4]:  # Fluoride exposure
        st.write("- You have limited or no exposure to fluoride. Consider using fluoride toothpaste or other fluoride treatments.")
    
    if input_data[0][5]:  # Dry mouth
        st.write("- You have dry mouth. This can increase your caries risk by reducing natural tooth protection.")
    
    if input_data[0][6] > 2:  # Frequent snacking
        st.write("- You frequently snack. Try to limit snacking between meals to reduce caries risk.")
    
    if input_data[0][7] < 2:  # Dental visits
        st.write("- Your dental visits are irregular. Regular check-ups are crucial for preventing caries.")
    
    if input_data[0][8] and input_data[0][0] <= 5:  # Nighttime bottle feeding
        st.write("- Nighttime bottle feeding increases caries risk. Try to avoid bottle feeding at night.")
    
    if input_data[0][9]:  # Orthodontic appliances
        st.write("- You have orthodontic appliances (e.g., braces). These can make cleaning harder and increase caries risk.")
    
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
    
    # Generate new synthetic data
    data, labels = generate_synthetic_data()
    
    # Check if model exists and has the correct number of features
    if (not os.path.exists('caries_model.joblib') or 
        not os.path.exists('scaler.joblib') or
        joblib.load('scaler.joblib').n_features_in_ != data.shape[1]):
        
        st.write("Training new model...")
        model, scaler = train_model((data, labels))
        joblib.dump(model, 'caries_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
    else:
        model = joblib.load('caries_model.joblib')
        scaler = joblib.load('scaler.joblib')
    
    st.write("Enter patient information:")
    
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    sugary_diet = st.slider("Diet high in sugary foods and beverages (0-10)", 0, 10, 5)
    brushing_frequency = st.selectbox("Brushing Frequency (times per day)", [0, 1, 2, 3])
    flossing_frequency = st.selectbox("Flossing Frequency (times per day)", [0, 1, 2])
    fluoride_exposure = st.checkbox("Limited or no exposure to fluoride")
    dry_mouth = st.checkbox("Do you experience dry mouth?")
    snacking_frequency = st.slider("Frequent Snacking (times per day)", 0, 10, 2)
    dental_visits = st.selectbox("Number of Dental Visits per Year", [0, 1, 2, 3, 4])
    
    nighttime_bottle_feeding = False
    if age <= 5:
        nighttime_bottle_feeding = st.checkbox("Nighttime bottle feeding")
    
    orthodontic_appliances = st.checkbox("Do you have orthodontic appliances (e.g., braces)?")
    
    if st.button("Predict Caries Risk"):
        input_data = np.array([[age, sugary_diet, brushing_frequency, flossing_frequency,
                                int(not fluoride_exposure), int(dry_mouth), snacking_frequency, 
                                dental_visits, int(nighttime_bottle_feeding), int(orthodontic_appliances)]])
        
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