import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_heartrate = data["le_heartrate"]
le_bloodoxygen = data["le_bloodoxygen"]
le_weight = data["le_weight"]
le_MRI = data["le_MRI"]
le_age = data["le_age"]
le_education = data["le_education"]
le_handdom = data["le_handdom"]
le_gender = data["le_gender"]
le_famhist = data["le_famhist"]
le_smoking = data["le_smoking"]
le_physicalact = data["le_physicalact"]
le_depression = data["le_depression"]
le_medication = data["le_medication"]
le_nutrition = data["le_nutrition"]
le_sleep = data["le_sleep"]





def show_predict_page():
    st.title("Software Developer Diabetes Prediction")

    st.write("""### We need some information to predict the risk""")


    educationlvl = (
        "No School",
        "Primary School",
        "Secondary School",
        "Diploma/Degree",
    )
    handdom = (
        "Right",
        "Left",
    )
    famhistor = (
        "Yes",
        "No",
    )
    medication = (
        "Yes",
        "No",
    )

    smoking = (
        "Current Smoker",
        "Former Smoker",
        "Never Smoked",
    )

    gender = (
        "Male",
        "Female",
    )

    physicalact = (
        "Sedentary",
        "Moderate Activity",
        "Mild Activity",
    )

    depression = (
        "Yes",
        "No",
    )

    nutrition = (
        "Balanced Diet",
        "Mediterranean Diet",
        "Low-Carb Diet",
    )

    sleep = (
        "Good",
        "Poor",
    )


    smoke = st.selectbox("Smoking", smoking)
    gendr = st.selectbox("Gender", gender)
    activity = st.selectbox("Activeness", physicalact)
    depress = st.selectbox("Depression", depression)
    diet = st.selectbox("Diet", nutrition)
    sleephealth = st.selectbox("Sleep Health", sleep)
    
    educationlvl = st.selectbox("Education Level", educationlvl)
    handdom = st.selectbox("Hand dominance", handdom)
    famhistor = st.selectbox("Fam History", famhistor)
    medication = st.selectbox("Medication", medication)
    
    hrtrate = st.slider("Heart Rate", 0, 150, 10)
    bloodoxy = st.slider("Blood Oxygen Level", 0, 150, 10)
    weight = st.slider("Weight", 0, 300, 50)
    ages = st.slider("Age", 0, 100, 1)
    mrinumber = st.slider("MRI", 0, 3, .1)

    ok = st.button("Calculate Risk")
    if ok:
        X = np.array([[ hrtrate, bloodoxy, weight, mrinumber, ages, educationlvl, handdom, gendr, famhistor, smoke, activity, depress, medication, diet, sleephealth]])
        X.iloc[:, 0] = le_heartrate.transform(X.iloc[:, 0])
        X.iloc[:, 1] = le_bloodoxygen.transform(X.iloc[:, 1])
        X.iloc[:, 3] = le_weight.transform(X.iloc[:, 3])
        X.iloc[:, 4] = le_MRI.transform(X.iloc[:, 4])
        X.iloc[:, 5] = le_age.transform(X.iloc[:, 5])
        X.iloc[:, 6] = le_education.transform(X.iloc[:, 6])
        X.iloc[:, 7] = le_handdom.transform(X.iloc[:, 7])
        X.iloc[:, 8] = le_gender.transform(X.iloc[:, 8])
        X.iloc[:, 9] = le_famhist.transform(X.iloc[:, 9])
        X.iloc[:, 10] = le_smoking.transform(X.iloc[:, 10])
        X.iloc[:, 11] = le_physicalact.transform(X.iloc[:, 11])
        X.iloc[:, 12] = le_depression.transform(X.iloc[:, 12])
        X.iloc[:, 13] = le_medication.transform(X.iloc[:, 13])
        X.iloc[:, 14] = le_nutrition.transform(X.iloc[:, 14])
        X.iloc[:, 15] = le_sleep.transform(X.iloc[:, 15])
        X = X.astype(float)

        risk = regressor.predict(X)
        st.subheader(f"The estimated salary is {risk[0]:.2f}")
