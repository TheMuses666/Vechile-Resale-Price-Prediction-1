import streamlit as st
import numpy as np
import pandas as pd

from read_file import read_files

model, feature_columns=read_files()

st.title('Car Price Prediction Model')
st.write('Enter the Car Detail here')

condition_map = {
    "Used": 0,
    "New": 1
}

vehicle_condition_label = st.selectbox(
    "Vehicle Condition",
    list(condition_map.keys())
)

vehicle_condition = condition_map[vehicle_condition_label]

if vehicle_condition == 1:
    mileage = 0
    st.number_input("Mileage", value=0, disabled=True)
    car_age = 0
    st.number_input('Car Age',value=0, disabled=True)
else:
    mileage = st.slider("Mileage", 0, 200000, 80000)
    car_age = st.number_input("Car Age", value=5)

crossover_map = {
    "No": 0,
    "Yes": 1
}

crossover_label = st.selectbox(
    "Crossover Car and Van",
    list(crossover_map.keys())
)

crossover = crossover_map[crossover_label]


color = st.selectbox(
    'Color:',
    [
        "Black","Blue","Bronze","Brown","Burgundy","Gold","Green","Grey",
        "Indigo","Magenta","Maroon","Multicolour","Navy","Orange","Pink",
        "Purple","Red","Silver","Turquoise","White","Yellow"
    ]
)

body_type = st.selectbox(
    'Body Type:',
    [
        "Car Derived Van","Chassis Cab","Combi Van","Convertible","Coupe",
        "Estate","Hatchback","Limousine","MPV","Minibus","Panel Van",
        "Pickup","SUV","Saloon","Window Van"
    ]
)

fuel_type = st.selectbox(
    'Fuel Type:',
    [
        "Diesel","Diesel Hybrid","Diesel Plug-in Hybrid","Electric",
        "Natural Gas","Petrol","Petrol Hybrid","Petrol Plug-in Hybrid"
    ]
)

if st.button('Price Predict'):
    sample = pd.DataFrame(0,index=[0],columns=feature_columns)

    sample.loc[0, "mileage"] = mileage
    sample.loc[0, "vehicle_condition"] = vehicle_condition
    sample.loc[0, "crossover_car_and_van"] = crossover
    sample.loc[0, "car_age"] = car_age

    colour_col = f"standard_colour_{color}"
    body_col = f"body_type_{body_type}"
    fuel_col = f"fuel_type_{fuel_type}"

    if colour_col in sample.columns:
        sample.loc[0, colour_col] = 1

    if body_col in sample.columns:
        sample.loc[0, body_col] = 1

    if fuel_col in sample.columns:
        sample.loc[0, fuel_col] = 1

    pred_log = model.predict(sample)[0]
    pred_price = np.exp(pred_log)

    st.metric(
    label="Predicted Price",
    value=f"£{pred_price:,.2f}",
    delta="Based on ML Random Forest model"
    )
