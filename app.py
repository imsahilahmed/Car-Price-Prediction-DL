import streamlit as st
import torch
import torch.nn as nn

# ---------- UI ----------
st.set_page_config(page_title="Car Price Predictor")
st.title("🚗 Car Price Prediction")

# ---------- Load Model ----------
checkpoint = torch.load("car_price_model.pth", map_location="cpu")

model = nn.Linear(3, 1)
model.load_state_dict(checkpoint["model_state"])
model.eval()

X_mean = checkpoint["X_mean"]
X_std  = checkpoint["X_std"]
y_mean = checkpoint["y_mean"]
y_std  = checkpoint["y_std"]

# ---------- User Inputs ----------
age = st.number_input("Car Age (years)", min_value=0, max_value=30)
mileage = st.number_input("Car Mileage", min_value=0)

accident = st.selectbox(
    "Is the car accident-free?",
    ["Yes", "No"]
)
accident_free = 1 if accident == "Yes" else 0

# ---------- Prediction ----------
if st.button("Predict Price"):
    X_input = torch.tensor(
        [[age, mileage, accident_free]],
        dtype=torch.float32
    )

    X_norm = (X_input - X_mean) / X_std

    with torch.no_grad():
        y_pred_norm = model(X_norm)

    y_pred = y_pred_norm * y_std + y_mean

    st.success(f"Estimated Car Price: ₹ {y_pred.item():,.2f}")
