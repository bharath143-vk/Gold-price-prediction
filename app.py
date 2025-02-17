import streamlit as st
import numpy as np

w_final=np.load('weights.npy')
b_final=np.load('bias.npy')

scaler_mean=np.load('scaler_mean.npy')
scaler_scale=np.load('scaler_scale.npy')


st.title("Gold Price Prediction")
spx=st.number_input("Enter SPX")
uso=st.number_input("Enter USO")
slv=st.number_input("Enter SLV")
eur_usd=st.number_input("Enter EUR/USD")

if st.button("Predict Gold Price"):
    # Ensure at least one input is non-zero before prediction
    if spx != 0.0 or uso != 0.0 or slv != 0.0 or eur_usd != 0.0:
        # Prepare input features as an array
        features = np.array([spx, uso, slv, eur_usd])
        #standardize using the same mean and scale from training
        standardized_features=(features-scaler_mean)/scaler_scale
        
        # Calculate prediction
        prediction = np.dot(standardized_features, w_final) + b_final
        st.success(f"Predicted Gold Price: {prediction}")
    else:
        st.warning("Please enter at least one value to predict.")
