import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

st.title("Predict if customer will buy or not")

# load model with joblib
sc = load("scaler.pkl")
model = load("model.pkl")

 # get input from user
age = st.number_input("Age", min_value=0, max_value=100, value=25)
income = st.number_input("Estimated Salary", min_value=0, max_value=1000000, value=50000)

# create form and predict 
if st.button("Predict"):

    # standadize and make prediction 
    prediction = model.predict(sc.transform(np.array([[age, income]])))
    if prediction == 1:
        st.write("Customer will buy")
    else:
        st.write("Sorry, Customer will not buy")