import streamlit as st
import pickle

# Load the trained model and label encoder
with open('./models/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('./models/contract_encoder.pkl', 'rb') as file:
    contract_encoder = pickle.load(file)
with open('./models/payment_method_encoder.pkl', 'rb') as file:
    payment_method_encoder = pickle.load(file)
with open('./models/internet_service_encoder.pkl', 'rb') as file:
    internet_service_encoder = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")
# Input fields for user to enter customer

features =['tenure',
           'MonthlyCharges',
           'TotalCharges',
           'Contract',
           'PaymentMethod',
           'InternetService']

tenure = st.text_input(label = "Enter tenure (in months):")
monthly_charges = st.text_input(label = "Enter monthly charges:")
total_charges = st.text_input(label = "Enter total charges:")
contract = st.selectbox("Select contract type:", options=["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Select payment method:", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet_service = st.selectbox("Select internet service type:", options=["DSL", "Fiber optic", "No"])  

if st.button("Predict Churn"):
    contract_encoded = contract_encoder.transform([contract])[0]
    payment_method_encoded = payment_method_encoder.transform([payment_method])[0]
    internet_service_encoded = internet_service_encoder.transform([internet_service])[0]
    input_data = [[int(tenure), float(monthly_charges), float(total_charges), contract_encoded, payment_method_encoded, internet_service_encoded]]
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}")

