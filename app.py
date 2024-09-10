import streamlit as st
import pickle
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import numpy as np
import pandas as pd

# Load the model
with open("diabetes_2024.pkcls", "rb") as model:
    loaded_model = pickle.load(model)

header = st.container()
body = st.container()

# Define function to process patient details
def patient_details(normalized_age, patient_gender, patient_polyuria, patient_polydipsia, patient_sudden_weight_loss, patient_weakness, patient_polyphagia, patient_genital_thrush, patient_visual_blurring, patient_itching, patient_irritablity, patient_delayed_healing, patient_partial_paresis, patient_muscle_stiffness, patient_alopecia, patient_obesity):

    domain = Domain([ContinuousVariable('Age'),
                     DiscreteVariable('Gender', values=["Female", "Male"]),
                     DiscreteVariable('Polyuria', values=["No", "Yes"]),
                     DiscreteVariable('Polydipsia', values=["No", "Yes"]),
                     DiscreteVariable('Sudden Weight Loss', values=["No", "Yes"]),
                     DiscreteVariable('Weakness', values=["No", "Yes"]),
                     DiscreteVariable('Polyphagia', values=["No", "Yes"]),
                     DiscreteVariable('Genital Thrush', values=["No", "Yes"]),
                     DiscreteVariable('Visual Blurring', values=["No", "Yes"]),
                     DiscreteVariable('Itching', values=["No", "Yes"]),
                     DiscreteVariable('Irritability', values=["No", "Yes"]),
                     DiscreteVariable('Delayed Healing', values=["No", "Yes"]),
                     DiscreteVariable('Partial Paresis', values=["No", "Yes"]),
                     DiscreteVariable('Muscle Stiffness', values=["No", "Yes"]),
                     DiscreteVariable('Alopecia', values=["No", "Yes"]),
                     DiscreteVariable('Obesity', values=["No", "Yes"])])

    # Encode categorical variables as integers
    gender = ["Female", "Male"]
    gender_val = gender.index(patient_gender)

    # Convert input fields to numeric values for model input
    polyuria_val = ["No", "Yes"].index(patient_polyuria)
    polydipsia_val = ["No", "Yes"].index(patient_polydipsia)
    sudden_weight_loss_val = ["No", "Yes"].index(patient_sudden_weight_loss)
    weakness_val = ["No", "Yes"].index(patient_weakness)
    polyphagia_val = ["No", "Yes"].index(patient_polyphagia)
    genital_thrush_val = ["No", "Yes"].index(patient_genital_thrush)
    visual_blurring_val = ["No", "Yes"].index(patient_visual_blurring)
    itching_val = ["No", "Yes"].index(patient_itching)
    irritablity_val = ["No", "Yes"].index(patient_irritablity)
    delayed_healing_val = ["No", "Yes"].index(patient_delayed_healing)
    partial_paresis_val = ["No", "Yes"].index(patient_partial_paresis)
    muscle_stiffness_val = ["No", "Yes"].index(patient_muscle_stiffness)
    alopecia_val = ["No", "Yes"].index(patient_alopecia)
    obesity_val = ["No", "Yes"].index(patient_obesity)

    # Combine all input data into a structured format
    user_input_values = (normalized_age, gender_val, polyuria_val, polydipsia_val, sudden_weight_loss_val, weakness_val, polyphagia_val, genital_thrush_val, visual_blurring_val, itching_val, irritablity_val, delayed_healing_val, partial_paresis_val, muscle_stiffness_val, alopecia_val, obesity_val)
    
    X = np.column_stack(user_input_values)
    data_table = Orange.data.Table(domain, X)

    return data_table

# Header container for the app title and author information
with header:
    st.markdown("<h1 style='text-align: center; color: #FFA07A;'>Diabetes Prediction using Orange 3</h1>", unsafe_allow_html=True)
    st.header("By Dr. Atul Tiwari")
    st.divider()

# Body container for user input and prediction result
with body:
    st.markdown("### Enter the patient’s details")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Slider for Age input
        patient_age = st.slider('Age', 16, 90, 45)
        normalized_age = (patient_age - 16) / (90 - 16)

        # Gender input as radio buttons
        patient_gender = st.radio("Gender", ("Female", "Male"))
        
        # Symptom inputs in the first column
        patient_polyuria = st.radio("Polyuria", ("No", "Yes"))
        patient_polydipsia = st.radio("Polydipsia", ("No", "Yes"))
        patient_sudden_weight_loss = st.radio("Sudden Weight Loss", ("No", "Yes"))
        patient_weakness = st.radio("Weakness", ("No", "Yes"))
        patient_polyphagia = st.radio("Polyphagia", ("No", "Yes"))

    with col2:
        # Additional symptom inputs in the second column
        patient_genital_thrush = st.radio("Genital Thrush", ("No", "Yes"))
        patient_visual_blurring = st.radio("Visual Blurring", ("No", "Yes"))
        patient_itching = st.radio("Itching", ("No", "Yes"))
        patient_irritablity = st.radio("Irritability", ("No", "Yes"))
        patient_delayed_healing = st.radio("Delayed Healing", ("No", "Yes"))
        patient_partial_paresis = st.radio("Partial Paresis", ("No", "Yes"))
        patient_muscle_stiffness = st.radio("Muscle Stiffness", ("No", "Yes"))
        patient_alopecia = st.radio("Alopecia", ("No", "Yes"))
        patient_obesity = st.radio("Obesity", ("No", "Yes"))

    # Get patient data for prediction
    patient_data = patient_details(normalized_age, patient_gender, patient_polyuria, patient_polydipsia, patient_sudden_weight_loss, patient_weakness, patient_polyphagia, patient_genital_thrush, patient_visual_blurring, patient_itching, patient_irritablity, patient_delayed_healing, patient_partial_paresis, patient_muscle_stiffness, patient_alopecia, patient_obesity)

    st.divider()

    # Prediction section
    st.markdown("## Prediction")

    # Show spinner while calculating the prediction
    with st.spinner("Calculating..."):
        prediction = loaded_model(patient_data)
    
    # Display prediction result with appropriate styling
    if prediction:
        st.success("#### Patient is likely to have Diabetes")
    else:
        st.error("#### Patient is **NOT** likely to have Diabetes")

    st.divider()
