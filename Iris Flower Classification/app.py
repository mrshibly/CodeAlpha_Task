import pickle
import streamlit as st
import numpy as np

with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Iris Flower Class Predictor")

st.write("Choose the measurements for Sepal and Petal dimensions from the dropdowns below:")

sepal_length = st.selectbox("Sepal Length (cm)", [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0], index=4)
sepal_width = st.selectbox("Sepal Width (cm)", [2.0, 2.5, 3.0, 3.5, 4.0, 4.5], index=2)
petal_length = st.selectbox("Petal Length (cm)", [1.0, 2.5, 3.0, 4.5, 5.0, 6.0, 7.0], index=3)
petal_width = st.selectbox("Petal Width (cm)", [0.1, 0.5, 1.0, 1.5, 2.0, 2.5], index=2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    st.write(f"### Predicted Class: {prediction[0]}")
    st.write("### Prediction Probabilities:")
    st.write({
        "Setosa": prediction_proba[0][0],
        "Versicolor": prediction_proba[0][1],
        "Virginica": prediction_proba[0][2],
    })
