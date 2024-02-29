import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


@st.cache
def load_data():
    df = pd.read_csv("Resources/dementia_patients_health_data.csv")
    df = df[["Diabetic", "HeartRate", "BloodOxygenLevel", "BodyTemperature", "Weight", "MRI_Delay", "Age", "Education_Level",  "Dominant_Hand", "Gender", "Family_History", "Smoking_Status",  "Physical_Activity", "Depression_Status", "Medication_History", "Nutrition_Diet", "Sleep_Quality"]]
    df.info()
    df = df.dropna()
    df.isnull().sum()
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Software Engineer Risks")

    st.write(
        """
    ### Stack Overflow Developer Survey 2020
    """
    )

    data = df["Physical_Activity"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)
    
    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Weight"])["Nutrition_Diet"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["Age"])["Sleep_Quality"].mean().sort_values(ascending=True)
    st.line_chart(data)

