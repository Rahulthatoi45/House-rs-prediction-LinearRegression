# 🏠 House Price Prediction

A Machine Learning project that predicts the estimated price of a house based on its features such as area, number of bedrooms, bathrooms, floors, house age, garden availability, and location.

The project uses **Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Streamlit** to build and deploy an interactive house-price prediction application.

---

## 📌 Project Overview

House prices depend on several factors, including property size, number of rooms, location, age of the property, and additional facilities.

In this project, a **Machine Learning regression model** is trained to learn the relationship between these features and the house price.

The trained model is then integrated with a **Streamlit web application**, where users can enter house details and get an estimated price.

---

## 🎯 Objective

The main objectives of this project are:

- Analyze a house-price dataset.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the data.
- Convert categorical features into numerical values.
- Train a Machine Learning regression model.
- Evaluate the model using regression metrics.
- Save the trained model.
- Build an interactive Streamlit application for prediction.

---

## 📊 Features Used

The model uses the following features:

| Feature | Description |
|---|---|
| `Area_sqft` | Area of the house in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Floors` | Number of floors |
| `HouseAge` | Age of the house in years |
| `HasGarden` | Whether the house has a garden |
| `Location` | Location category of the house |

### Target Variable

The target variable is:

```text
Price
