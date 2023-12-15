from flask import Flask, jsonify, request
import pandas as pd
import pickle
import torch

torch.set_num_threads(1)

app = Flask(__name__)

# import LGBM classifier
LGBMclassifier = open("lgbm_classifier.pkl", "rb")
classifier = pickle.load(LGBMclassifier)

# import df test
app_test = pd.read_csv('df_merged_test_reduced.csv')

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>Bienvenue sur l'API !</h1>
    <p> Pour retrouver les identifiants clients : <a href="https://juguirlet.pythonanywhere.com/api/v1/customers">Cliquez-ici</a></p>
    <p> Pour retrouver les caractéristiques d'un client : <a href="https://juguirlet.pythonanywhere.com/api/v1/customers/100001">Cliquez sur ce lien et modifiez le n° dans l'URL en remplaçant par l'identifiant client de votre choix</a></p>
    <p> Pour afficher la probabilité du risque de crédit pour un client : <a href="https://juguirlet.pythonanywhere.com/api/v1/customers/100001/pred_score">Cliquez sur ce lien et modifiez le n° dans l'URL en remplaçant par l'identifiant client de votre choix</a></p>
    '''

# get the customers id
@app.route('/api/v1/customers', methods=['GET'])
def customers_ids():
    customers = app_test['SK_ID_CURR'].tolist()
    return jsonify(customers)

# get the features values for a customer
@app.route('/api/v1/customers/<int:customer_id>', methods=['GET'])
def columns_values(customer_id: int):
    if not app_test['SK_ID_CURR'].isin([customer_id]).any():
        return "Error: No id field provided. Please specify an id"
    content = app_test[app_test['SK_ID_CURR'] == customer_id].iloc[0].to_dict()
    return jsonify(content)

# example of endpoint on the api to return predicted class
@app.route('/api/v1/customers/<int:customer_id>/pred_score', methods=['GET'])
def predict_customer(customer_id: int):
    if not app_test['SK_ID_CURR'].isin([customer_id]).any():
        return "Error: No id field provided. Please specify an id"
    proba = classifier.predict_proba(app_test.loc[app_test['SK_ID_CURR'] == customer_id,app_test.columns[:-2]])
    proba_risk = (proba[:,1]*100).item()
    proba_risk_str = f"Probabilité que le crédit soit refusé : {proba_risk:.2f}%"
    return proba_risk_str

# function to calculate proba with values selected on the app
def predict_function(data):
    # Get feature names and values from the input data
    inner_data = data.get('data', {})
    feature_names = list(inner_data.keys())
    feature_values = list(inner_data.values())
    # Create a DataFrame with feature names as columns
    input_data_df = pd.DataFrame([feature_values], columns=feature_names)
    print(input_data_df)
    proba = classifier.predict_proba(input_data_df).tolist()
    return proba

#define a route for only post request
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    #getting an array of features from the post request's body
    data = request.get_json()
    print(data)

    prediction = predict_function(data)
    # Return the prediction result
    result = {
        "prediction": prediction,
        "message": "Prediction successful"
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)