# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:26:45 2025
@author: adity
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Load the model
model = pickle.load(open('C:/Users/adity/OneDrive/Desktop/fraud_detection_app/trained_model.sav', 'rb'))

# Feature names
columns = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

#  Prediction function
def fraud_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(input_df)
    return prediction[0]  # return the class (0 or 1)

#  Streamlit UI
st.title("Credit Card Fraud Detection App")

st.write("Enter the transaction details:")

# Taking input from user
user_input = []
for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

# Predict on button click
if st.button("Check for Fraud"):
    result = fraud_prediction(user_input)
    if result == 0:
        st.success(" This transaction is **not fraudulent**.")
    else:
        st.error(" This transaction is **fraudulent**!")

