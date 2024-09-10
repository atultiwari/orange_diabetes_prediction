import streamlit as st
import pickle
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import numpy as np
import pandas as pd

with open("diabetes_2024", "rb") as model:
    loaded_model = pickle.load(model)

header = st.container()
body = st.container()

def patient_details(normalized_age, patient_gender, patient_polyuria, patient_polydipsia, patient_sudden_weight_loss, patient_weakness, patient_polyphagia, patient_genital_thrush, patient_visual_blurring, patient_itching, patient_irritablity, patient_delayed_healing, patient_partial_paresis, patient_muscle_stiffness, patient_alopecia, patient_obesity):

    domain = Domain([ContinuousVariable('Age'), DiscreteVariable('Gender', values=["Female", "Male"]), DiscreteVariable('Polyuria', values=["No", "Yes"]), DiscreteVariable('Polydipsia', values=["No", "Yes"]), DiscreteVariable('sudden weight loss', values=["No", "Yes"]), DiscreteVariable('weakness', values=["No", "Yes"]), DiscreteVariable('Polyphagia', values=["No", "Yes"]), DiscreteVariable('Genital thrush', values=["No", "Yes"]), DiscreteVariable('visual blurring', values=["No", "Yes"]), DiscreteVariable('Itching', values=["No", "Yes"]), DiscreteVariable('Irritability', values=["No", "Yes"]), DiscreteVariable('delayed healing', values=["No", "Yes"]), DiscreteVariable('partial paresis', values=["No", "Yes"]), DiscreteVariable('muscle stiffness', values=["No", "Yes"]), DiscreteVariable('Alopecia', values=["No", "Yes"]), DiscreteVariable('Obesity', values=["No", "Yes"])])

    gender = ["Female", "Male"]
    gender_val = gender.index(patient_gender)

    polyuria = polydipsia = sudden_weight_loss = weakness = polyphagia = genital_thrush = visual_blurring = itching = irritablity = delayed_healing = partial_paresis = muscle_stiffness = alopecia = obesity = ["No", "Yes"]

    polyuria_val = polyuria.index(patient_polyuria)
    polydipsia_val = polydipsia.index(patient_polydipsia)
    sudden_weight_loss_val = sudden_weight_loss.index(patient_sudden_weight_loss)
    weakness_val = weakness.index(patient_weakness)
    polyphagia_val = polyphagia.index(patient_polyphagia)
    genital_thrush_val = genital_thrush.index(patient_genital_thrush)
    visual_blurring_val = visual_blurring.index(patient_visual_blurring)
    itching_val = itching.index(patient_itching)
    irritablity_val = irritablity.index(patient_irritablity)
    delayed_healing_val = delayed_healing.index(patient_delayed_healing)
    partial_paresis_val = partial_paresis.index(patient_partial_paresis)
    muscle_stiffness_val = muscle_stiffness.index(patient_muscle_stiffness)
    alopecia_val = alopecia.index(patient_alopecia)
    obesity_val = obesity.index(patient_obesity)

    user_input_values = (normalized_age, gender_val, polyuria_val, polydipsia_val, sudden_weight_loss_val, weakness_val, polyphagia_val,genital_thrush_val, visual_blurring_val, itching_val, irritablity_val, delayed_healing_val, partial_paresis_val, muscle_stiffness_val, alopecia_val, obesity_val)


    X = np.column_stack(user_input_values)
    data_table = Orange.data.Table(domain, X)

    return data_table

with header:
    st.title("Diabetes Prediction using Orange 3")
    st.header("By Dr. Atul Tiwari")
    st.divider()

with body:
    st.markdown("### Enter the patients details")
    # Patient details goes here ---
    patient_age = st.number_input('Age', step=1)
    # normalization = (variable_value - minimum_value)/ (max - min)
    normalized_age = (patient_age - 16) / (90 - 16)

    patient_gender = st.radio("Gender", ("Female", "Male"))
    patient_polyuria = st.radio("Polyuria", ("No", "Yes"))
    patient_polydipsia = st.radio("Polydipsia", ("No", "Yes"))
    patient_sudden_weight_loss = st.radio("Sudden Weight Loss", ("No", "Yes"))
    patient_weakness = st.radio("Weakness", ("No", "Yes"))
    patient_polyphagia = st.radio("Polyphagia", ("No", "Yes"))
    patient_genital_thrush = st.radio("Genital Thrush", ("No", "Yes"))
    patient_visual_blurring = st.radio("Visual Blurring", ("No", "Yes"))
    patient_itching = st.radio("Itching", ("No", "Yes"))
    patient_irritablity = st.radio("Irritablity", ("No", "Yes"))
    patient_delayed_healing = st.radio("Delayed Healing", ("No", "Yes"))
    patient_partial_paresis = st.radio("Partial Paresis", ("No", "Yes"))
    patient_muscle_stiffness = st.radio("Muscle Stiffness", ("No", "Yes"))
    patient_alopecia = st.radio("Alopecia", ("No", "Yes"))
    patient_obesity = st.radio("Obesity", ("No", "Yes"))


    patient_data = patient_details(normalized_age, patient_gender, patient_polyuria, patient_polydipsia, patient_sudden_weight_loss, patient_weakness, patient_polyphagia, patient_genital_thrush, patient_visual_blurring, patient_itching, patient_irritablity, patient_delayed_healing, patient_partial_paresis, patient_muscle_stiffness, patient_alopecia, patient_obesity)



    st.divider()
    st.markdown("## Prediction")
    prediction = loaded_model(patient_data)
    if prediction:
        st.markdown("#### Patient is likely to have Diabetes")
    else:
        st.markdown("#### Patient is **NOT** likely to have Diabetes")

    st.divider()
