import streamlit as st
import pickle
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import numpy as np
import pandas as pd

# Load the model
with open("diabetes_sample_idrs.pkcls", "rb") as model:
    loaded_model = pickle.load(model)

header = st.container()
body = st.container()

# Define function to process patient details for IDRS
def patient_details(age, waist_circumference, physical_activity, family_history):
    domain = Domain([ContinuousVariable('Age'),
                     ContinuousVariable('Waist Circumference'),
                     DiscreteVariable('Physical Activity', values=["Low", "Moderate", "High"]),
                     DiscreteVariable('Family History', values=["No", "Yes"])])

    # Encode categorical variables as integers
    activity = ["Low", "Moderate", "High"]
    activity_val = activity.index(physical_activity)

    family_history_val = 1 if family_history == "Yes" else 0

    # Convert input into appropriate format
    user_input_values = (age, waist_circumference, activity_val, family_history_val)
    
    X = np.column_stack(user_input_values)
    data_table = Orange.data.Table(domain, X)

    return data_table

# Header container for the app title and author information
with header:
    st.markdown("<h1 style='text-align: center; color: #FFA07A;'>Diabetes Risk Prediction using IDRS</h1>", unsafe_allow_html=True)
    st.header("By Dr. Atul Tiwari")
    st.divider()

# Body container for user input and prediction result
with body:
    st.markdown("### Enter the patient’s details")

    # Split the input fields into two columns for a better layout
    col1, col2 = st.columns(2)

    with col1:
        # Slider for Age input
        patient_age = st.slider('Age', 0, 120, 45)
        # Radio buttons for Physical Activity input
        patient_activity = st.radio("Physical Activity", ["Low", "Moderate", "High"])

    with col2:
        # Slider for Waist Circumference input
        patient_waist = st.slider('Waist Circumference (in cm)', 50, 150, 85)
        # Radio buttons for Family History of Diabetes input
        patient_family_history = st.radio("Family History of Diabetes", ["No", "Yes"])

    # Divider line before the prediction section
    st.divider()

    # Prediction section with a spinner for a better user experience
    st.markdown("## Prediction")
    
    # Show spinner while the model predicts
    with st.spinner("Calculating..."):
        patient_data = patient_details(patient_age, patient_waist, patient_activity, patient_family_history)
        prediction = loaded_model(patient_data)
        
    # Display prediction result
    if prediction:
        st.success("#### Patient is likely to have Diabetes")
    else:
        st.error("#### Patient is **NOT** likely to have Diabetes")

    # Divider line at the end
    st.divider()
