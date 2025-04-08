import streamlit as st
import numpy as np
import pickle

with open('Model/scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('Model/Model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)



st.title("E-Commerce Predictor")

avg_sessio_length = st.number_input("Avg. Session Length")

time_on_app = st.number_input("Time on App")

length_of_Membership = st.number_input("Length of Membership")

if st.button("Predict"):
    data = np.array([avg_sessio_length,time_on_app,length_of_Membership]).reshape(1,-1)

    data_new = loaded_scaler.transform(data)

    prediction = loaded_model.predict(data_new)
    
    st.success(f'Yearly Spend will be:  ${round(prediction[0], 2)}')

