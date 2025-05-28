import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Train model
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor (California Housing Data)")

st.write("Enter housing features below to predict the price:")

# User inputs
MedInc = st.slider("Median Income (10k $)", 0.0, 15.0, 5.0)
HouseAge = st.slider("House Age (years)", 1, 50, 20)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 1.0, 5.0, 1.0)
Population = st.slider("Population", 100, 5000, 1000)
AveOccup = st.slider("Average Occupancy", 1.0, 5.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

# Predict
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])
prediction = model.predict(input_data)[0]

st.subheader("📊 Predicted House Price:")
st.success(f"${prediction * 100000:.2f}")
