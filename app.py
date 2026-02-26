from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
# Load the trained model and label encoder
with open('./models/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('./models/contract_encoder.pkl', 'rb') as file:
    contract_encoder = pickle.load(file)
with open('./models/payment_method_encoder.pkl', 'rb') as file:
    payment_method_encoder = pickle.load(file)
with open('./models/internet_service_encoder.pkl', 'rb') as file:
    internet_service_encoder = pickle.load(file)    


@app.route('/',methods=['GET'])
def home():
    return "Welcome to the Customer Churn Prediction API! Use the /predict endpoint to get predictions."
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tenure = data['tenure']
    monthly_charges = data['monthly_charges']
    total_charges = data['total_charges']
    contract = data['contract']
    payment_method = data['payment_method']
    internet_service = data['internet_service']
    
    contract_encoded = contract_encoder.transform([contract])[0]
    payment_method_encoded = payment_method_encoder.transform([payment_method])[0]
    internet_service_encoded = internet_service_encoder.transform([internet_service])[0]
    
    input_data = [[int(tenure), float(monthly_charges), float(total_charges), contract_encoded, payment_method_encoded, internet_service_encoded]]
    prediction = model.predict(input_data)[0]
    
    return jsonify({'churn_prediction': 'Yes' if prediction == 1 else 'No'})


if __name__ == '__main__':
    app.run(debug=True)