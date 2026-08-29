import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# House Price Prediction - Streamlit App
# ---------------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------- Load dataset ----------
@st.cache_data
def load_data():
    return pd.read_csv("house_price_dataset(1).csv")


df = load_data()

# Use the same features selected in the notebook
features = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "HouseAge",
    "HasGarden",
    "Location"
]

target = "Price"

# ---------- Prepare data ----------
@st.cache_resource
def train_model(data):
    data = data[features + [target]].copy()

    # Encode categorical columns in the same way as the notebook
    garden_encoder = LabelEncoder()
    location_encoder = LabelEncoder()

    data["HasGarden"] = garden_encoder.fit_transform(data["HasGarden"])
    data["Location"] = location_encoder.fit_transform(data["Location"])

    X = data[features]
    y = data[target]

    # Fit scaler once and reuse it for prediction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler, garden_encoder, location_encoder


model, scaler, garden_encoder, location_encoder = train_model(df)

# ---------- Page ----------
st.title("🏠 House Price Prediction")
st.write("Enter the house details below to estimate its price.")

st.divider()

# ---------- Input section ----------
st.subheader("House Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=int(df["Area_sqft"].min()),
        max_value=int(df["Area_sqft"].max()),
        value=int(df["Area_sqft"].median()),
        step=1
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=int(df["Bedrooms"].min()),
        max_value=int(df["Bedrooms"].max()),
        value=int(df["Bedrooms"].median()),
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=int(df["Bathrooms"].min()),
        max_value=int(df["Bathrooms"].max()),
        value=int(df["Bathrooms"].median()),
        step=1
    )

    floors = st.number_input(
        "Floors",
        min_value=int(df["Floors"].min()),
        max_value=int(df["Floors"].max()),
        value=int(df["Floors"].median()),
        step=1
    )

with col2:
    house_age = st.number_input(
        "House Age (years)",
        min_value=int(df["HouseAge"].min()),
        max_value=int(df["HouseAge"].max()),
        value=int(df["HouseAge"].median()),
        step=1
    )

    has_garden = st.selectbox(
        "Has Garden?",
        garden_encoder.classes_
    )

    location = st.selectbox(
        "Location",
        location_encoder.classes_
    )

st.divider()

# ---------- Prediction ----------
if st.button("🔮 Predict House Price", use_container_width=True):

    garden_encoded = garden_encoder.transform([has_garden])[0]
    location_encoded = location_encoder.transform([location])[0]

    input_data = pd.DataFrame({
        "Area_sqft": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "HouseAge": [house_age],
        "HasGarden": [garden_encoded],
        "Location": [location_encoded]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"### Estimated House Price: ₹{prediction:,.0f}")

    st.caption(
        "The estimate is generated using Linear Regression trained on the "
        "house price dataset."
    )

# ---------- Dataset information ----------
with st.expander("📊 Dataset Information"):
    st.write(f"**Number of houses:** {len(df)}")
    st.write(f"**Features used:** {len(features)}")
    st.write(f"**Minimum price:** ₹{df['Price'].min():,.0f}")
    st.write(f"**Maximum price:** ₹{df['Price'].max():,.0f}")

    st.dataframe(
        df[features + [target]].head(10),
        use_container_width=True
    )
