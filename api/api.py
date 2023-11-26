from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import sklearn

app = Flask(__name__)

# import LGBM classifier
LGBMclassifier = open(r'c:\Users\guirletj\Desktop\Test_envir\Projet 7\OC_Credit_Score_Project\api\lgbm_classifier.pkl',"rb")
classifier = pickle.load(LGBMclassifier, encoding='utf-8')

# import df test
app_test = pd.read_csv(r'c:\Users\guirletj\Desktop\Test_envir\Projet 7\OC_Credit_Score_Project\df_merged_test_reduced.csv')

@app.route('/', methods=['GET'])
def home():
    return "<h1>Bienvenue sur l'API !</h1> <p>Présentation des résultats</p>"

# getring the customers id
@app.route('/api/v1/customers', methods=['GET'])
def customers_ids():
    customers = app_test['SK_ID_CURR'].tolist()
    return jsonify(customers)

# getting the features values for a customer
@app.route('/api/v1/customers/<int:customer_id>', methods=['GET'])
def columns_values(customer_id: int):
    if not app_test['SK_ID_CURR'].isin([customer_id]).any():
        return "Error: No id field provided. Please specify an id"
    content = app_test[app_test['SK_ID_CURR'] == customer_id].iloc[0].to_dict()
    return jsonify(content)

# example of endpoint on the api to return predicted class
@app.route('/api/v1/customers/<int:customer_id>/predict', methods=['GET'])
def predict_customer(customer_id: int):
    if not app_test['SK_ID_CURR'].isin([customer_id]).any():
        return "Error: No id field provided. Please specify an id"
    predicted_class = app_test.loc[app_test['SK_ID_CURR'] == customer_id,'Predicted_Class'].tolist()
    return predicted_class

# function to calculate proba with values selected on the app
def predict_function(data):
    input_data = np.array(list(data.values())).reshape(1, -1)
    proba = classifier.predict_proba(input_data).tolist()
    return proba

#defining a route for only post request
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    #getting an array of features from the post request's body
    data = request.get_json()
    
    prediction = predict_function(data)
    # Return the prediction result
    result = {
        "prediction": prediction,
        "message": "Prediction successful"
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, threaded=True, debug=True)