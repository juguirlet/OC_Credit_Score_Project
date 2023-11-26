import streamlit as st
import numpy as np
from PIL import Image
import requests
import json
#import matplotlib.pyplot as plt
import shap
import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler

# import LGBM classifier
LGBMclassifier = open(r"C:\Users\guirletj\Desktop\Test_envir\Projet 7\models\lgbm_classifier.pkl", "rb")
classifier = pickle.load(LGBMclassifier)

# import df test
app_test = pd.read_csv(r'c:\Users\guirletj\Desktop\Test_envir\Projet 7\df_merged_test_reduced.csv')

# api url
api_url = "http://127.0.0.1:8000"

# functions to get the results from the api
def get_customers_ids():
    # list of customers ids
    customers_ids = requests.get(api_url + "/api/v1/customers")
    content = json.loads(customers_ids.content.decode('utf-8'))
    return content

def get_customer_values(customer_id):
    # list of parameters values for selected customer
    values_customer_id = requests.get(api_url + f"/api/v1/customers/{customer_id}")
    content = json.loads(values_customer_id.content.decode('utf-8'))
    return content

def get_features_names():
    features_names_list = app_test.columns.tolist()
    features_names_list.remove('SK_ID_CURR')
    features_names_list.remove('Predicted_Class')
    return features_names_list

features_names = get_features_names()
robust_scaler = RobustScaler()
robust_scaler.fit(app_test[features_names])

def get_customer_shap_values(data):
    scaled_data = robust_scaler.transform([data])
    customer_values_array = np.array(data).reshape(1, -1)
    explainer = shap.TreeExplainer(classifier.steps[-1][1])
    shap_values = explainer.shap_values(customer_values_array)
    features_names = get_features_names()
    return shap_values, customer_values_array, features_names

def get_predicted_score(): #valeurs des variables
    predicted_score_customer_id = requests.get(api_url + "/api/v1/customers/id/predict")
    content = json.loads(predicted_score_customer_id.content.decode('utf-8'))
    return content

def request_prediction(api_url_calc, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=api_url_calc, json=data_json, timeout=10)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def construire_jauge_score(score_remboursement_client):
    # Define the gauge ranges and colors
    gauge_ranges = [0, 0.55, 1]
    gauge_colors = ["#3C8B4E","#E0162B"]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 1))

    # Plot the gauge ranges with colors
    for i in range(1, len(gauge_ranges)):
        ax.fill_betweenx([0, 1], gauge_ranges[i - 1], gauge_ranges[i], color=gauge_colors[i - 1])

    # Plot the current value on the gauge
    ax.plot([score_remboursement_client, score_remboursement_client], [0, 1], color="black", linewidth=2)
    ax.plot([0.55,0.55], [0, 1], color="#A31D2B", linewidth=2, linestyle='--')

    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_title("Probabilité de remboursement du crédit")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(0, 1.1, 0.1)])
    ax.set_yticks([])

    return fig

customers_ids = get_customers_ids()
st.sidebar.markdown('<p style="font-family: San Francisco, sans-serif; font-size: 16px; color: darkblue; font-weight: bold;">'
                    'Sélectionner le client à tester :</p>', unsafe_allow_html=True)
customer_id = st.sidebar.selectbox('',customers_ids)
api_url_customer = f'http://127.0.0.1:8000/api/v1/customers/{customer_id}'
customer_data = get_customer_values(customer_id)

st.sidebar.markdown('<p style="font-family: San Francisco, sans-serif; font-size: 15px; color: darkblue;">'
                    'Variables du client non modifiables:</p>', unsafe_allow_html=True)

# Get the corresponding customer value
app_features_values = {}
app_features_values['DAYS_BIRTH'] = customer_data.get('DAYS_BIRTH')
app_features_values['APPROVED_AMT_DOWN_PAYMENT_MAX'] = customer_data.get('APPROVED_AMT_DOWN_PAYMENT_MAX')
app_features_values['APPROVED_CNT_PAYMENT_MEAN'] = customer_data.get('APPROVED_CNT_PAYMENT_MEAN')
app_features_values['EXT_SOURCE_2'] = customer_data.get('EXT_SOURCE_2')
app_features_values['EXT_SOURCE_3'] = customer_data.get('EXT_SOURCE_3')
app_features_values['INSTAL_AMT_PAYMENT_MIN'] = customer_data.get('INSTAL_AMT_PAYMENT_MIN')
app_features_values['INSTAL_AMT_PAYMENT_SUM'] = customer_data.get('INSTAL_AMT_PAYMENT_SUM')
app_features_values['INSTAL_DAYS_ENTRY_PAYMENT_MAX'] = customer_data.get('INSTAL_DAYS_ENTRY_PAYMENT_MAX')
app_features_values['INSTAL_DAYS_ENTRY_PAYMENT_MEAN'] = customer_data.get('INSTAL_DAYS_ENTRY_PAYMENT_MEAN')
app_features_values['INSTAL_DAYS_ENTRY_PAYMENT_SUM'] = customer_data.get('INSTAL_DAYS_ENTRY_PAYMENT_SUM')
app_features_values['INSTAL_DBD_MEAN'] =customer_data.get('INSTAL_DBD_MEAN')
app_features_values['PREV_AMT_ANNUITY_MEAN'] = customer_data.get('PREV_AMT_ANNUITY_MEAN')
app_features_values['PREV_APP_CREDIT_PERC_MEAN'] = customer_data.get('PREV_APP_CREDIT_PERC_MEAN')
app_features_values['PREV_CNT_PAYMENT_MEAN'] = customer_data.get('PREV_CNT_PAYMENT_MEAN')


st.sidebar.markdown('<p style="font-family: San Francisco, sans-serif; font-size: 12px;color: grey;">'
                    f'<strong>DAYS_BIRTH:</strong> {round(app_features_values["DAYS_BIRTH"], 2)}</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>APPROVED_AMT_DOWN_PAYMENT_MAX:</strong> {round(app_features_values["APPROVED_AMT_DOWN_PAYMENT_MAX"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>APPROVED_CNT_PAYMENT_MEAN:</strong> {round(app_features_values["APPROVED_CNT_PAYMENT_MEAN"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>EXT_SOURCE_2:</strong> {round(app_features_values["EXT_SOURCE_2"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>EXT_SOURCE_3:</strong> {round(app_features_values["EXT_SOURCE_3"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>INSTAL_AMT_PAYMENT_MIN:</strong> {round(app_features_values["INSTAL_AMT_PAYMENT_MIN"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>INSTAL_AMT_PAYMENT_SUM:</strong> {round(app_features_values["INSTAL_AMT_PAYMENT_SUM"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>INSTAL_DAYS_ENTRY_PAYMENT_MAX:</strong> {round(app_features_values["INSTAL_DAYS_ENTRY_PAYMENT_MAX"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>INSTAL_DAYS_ENTRY_PAYMENT_MEAN:</strong> {round(app_features_values["INSTAL_DAYS_ENTRY_PAYMENT_MEAN"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>INSTAL_DAYS_ENTRY_PAYMENT_SUM:</strong> {round(app_features_values["INSTAL_DAYS_ENTRY_PAYMENT_SUM"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>INSTAL_DBD_MEAN:</strong> {round(app_features_values["INSTAL_DBD_MEAN"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>PREV_AMT_ANNUITY_MEAN:</strong> {round(app_features_values["PREV_AMT_ANNUITY_MEAN"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>PREV_APP_CREDIT_PERC_MEAN:</strong> {round(app_features_values["PREV_APP_CREDIT_PERC_MEAN"], 2)}</p>', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="font-family: San Francisco, sans-serif; font-size: 12px; color: grey;">'
                    f'<strong>PREV_CNT_PAYMENT_MEAN:</strong> {round(app_features_values["PREV_CNT_PAYMENT_MEAN"], 2)}</p>', unsafe_allow_html=True)

st.title('Prédiction du score crédit')
st.subheader("Variables du client qui peuvent être modifiées")

# Create two columns
col1, col2 = st.columns(2)

app_features_values['ACTIVE_DAYS_CREDIT_MAX'] = col1.number_input('ACTIVE_DAYS_CREDIT_MAX',
                    value=customer_data.get('ACTIVE_DAYS_CREDIT_MAX'), step=1.0)
app_features_values['AMT_ANNUITY'] = col1.number_input('AMT_ANNUITY',
            value=customer_data.get('AMT_ANNUITY'), step=1.)
app_features_values['AMT_CREDIT'] = col1.number_input('AMT_CREDIT',
            value=customer_data.get('AMT_CREDIT'), step=1.)
app_features_values['AMT_GOODS_PRICE'] = col1.number_input('AMT_GOODS_PRICE',
            value=customer_data.get('AMT_GOODS_PRICE'), step=1.)
app_features_values['ANNUITY_INCOME_PERC'] = col1.number_input('ANNUITY_INCOME_PERC',
            value=customer_data.get('ANNUITY_INCOME_PERC'), step=0.1)
app_features_values['BURO_DAYS_CREDIT_MAX'] = col1.number_input('BURO_DAYS_CREDIT_MAX',
            value=customer_data.get('BURO_DAYS_CREDIT_MAX'), step=1.)
app_features_values['DAYS_EMPLOYED'] = col2.number_input('DAYS_EMPLOYED',
            value=customer_data.get('DAYS_EMPLOYED'), step=1.)
app_features_values['DAYS_EMPLOYED_PERC'] = col2.number_input('DAYS_EMPLOYED_PERC',
            value=customer_data.get('DAYS_EMPLOYED_PERC'), step=0.1)
app_features_values['DAYS_ID_PUBLISH'] = col2.number_input('DAYS_ID_PUBLISH',
            value=customer_data.get('DAYS_ID_PUBLISH'), step=1.)
app_features_values['PAYMENT_RATE'] = col2.number_input('PAYMENT_RATE',
            value=customer_data.get('PAYMENT_RATE'), step=0.1)
app_features_values['POS_COUNT'] = col2.number_input('POS_COUNT',
            value=customer_data.get('POS_COUNT'), step=1.)

predict_btn = st.button('Prédire')
if predict_btn:
    data = [app_features_values['PAYMENT_RATE'],
    app_features_values['EXT_SOURCE_2'],
    app_features_values['EXT_SOURCE_3'],
    app_features_values['DAYS_BIRTH'],
    app_features_values['DAYS_EMPLOYED'],
    app_features_values['AMT_ANNUITY'],
    app_features_values['ANNUITY_INCOME_PERC'],
    app_features_values['AMT_GOODS_PRICE'],
    app_features_values['INSTAL_AMT_PAYMENT_MIN'],
    app_features_values['ACTIVE_DAYS_CREDIT_MAX'],
    app_features_values['PREV_CNT_PAYMENT_MEAN'],
    app_features_values['APPROVED_CNT_PAYMENT_MEAN'],
    app_features_values['INSTAL_AMT_PAYMENT_SUM'],
    app_features_values['AMT_CREDIT'],
    app_features_values['INSTAL_DAYS_ENTRY_PAYMENT_MEAN'],
    app_features_values['POS_COUNT'],
    app_features_values['DAYS_ID_PUBLISH'],
    app_features_values['INSTAL_DAYS_ENTRY_PAYMENT_MAX'],
    app_features_values['DAYS_EMPLOYED_PERC'],
    app_features_values['PREV_AMT_ANNUITY_MEAN'],
    app_features_values['APPROVED_AMT_DOWN_PAYMENT_MAX'],
    app_features_values['INSTAL_DBD_MEAN'],
    app_features_values['INSTAL_DAYS_ENTRY_PAYMENT_SUM'],
    app_features_values['PREV_APP_CREDIT_PERC_MEAN'],
    app_features_values['BURO_DAYS_CREDIT_MAX']]
    api_url_calc = f'http://127.0.0.1:8000/api/v1/predict'
    pred = request_prediction(api_url_calc, data)
    prediction_list = pred.get("prediction", None)
    pred_score = prediction_list[0][1]

    if pred_score > 0.55:
        st.markdown("<p style='font-family: San Francisco, sans-serif; font-size:24px; color:red;'>Crédit refusé</p>",unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-family: San Francisco, sans-serif; font-size:24px; color:green;'>Crédit accordé</p>",unsafe_allow_html=True)
    st.write('Le risque de défaut pour ce client est de {:.2%}.'.format(pred_score))
    st.write('Le seuil de décision est de 55%.')
    jauge_score = construire_jauge_score(pred_score)
    st.pyplot(jauge_score)  
    with st.expander ("Voir les caractéristiques locales du client :"):
        shap_values, customer_values_array, features_names = get_customer_shap_values(data)
        st.set_option('deprecation.showPyplotGlobalUse', False)  # Suppress MatplotlibDeprecationWarning
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], customer_values_array, features_names, show=False)  # Use show=False to avoid double plotting
        st.pyplot(fig)
    
pred = None

feature_selected_1 = st.selectbox(
    'Sélectionner une 1re variable pour visualiser sa distribution selon le score du crédit',
    features_names)

def build_histogram(df, feature_selected_1, target):
    # Create a histogram using Matplotlib
    fig, ax = plt.subplots()

    # Filter data based on target
    df_target = app_test[app_test['Predicted_Class'] == target]
    customer_app = app_features_values[feature_selected_1]

    ax.hist(df_target[feature_selected_1], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    
    # Customize the plot
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel(feature_selected_1)
    ax.set_ylabel('Fréquence')
    if target==0:
        ax.set_title(f'Distribution de la variable {feature_selected_1} pour les clients ayant un crédit accordé')
    else:
        ax.set_title(f'Distribution de la variable {feature_selected_1} pour les clients ayant un crédit refusé')

    if customer_app is not None:
        ax.axvline(x=customer_app, color='red', linestyle='--', linewidth=2, label='Position du client')

    # Add legend
    ax.legend()
    return fig

# Create two columns
col1, col2 = st.columns(2)
histogram_target_0 = build_histogram(app_test, feature_selected_1, 0)
col1.pyplot(histogram_target_0)
histogram_target_1 = build_histogram(app_test, feature_selected_1, 1)
col2.pyplot(histogram_target_1)

feature_selected_2 = st.selectbox(
    'Sélectionner une 2e variable pour étudier les scores des clients selon les 2 variables sélectionnées',
    features_names)

def graph_two_features(feature_selected_1,feature_selected_2):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Separate data by class
    class_0 = app_test[app_test['Predicted_Class'] == 0]
    class_1 = app_test[app_test['Predicted_Class'] == 1]

    # Create a scatter plot with different colors for each class
    ax.scatter(class_0[feature_selected_1], class_0[feature_selected_2], color='#3C8B4E', label='Crédit accordé')
    ax.scatter(class_1[feature_selected_1], class_1[feature_selected_2], color='#E0162B', label='Crédit refusé')

    # Add a scatter plot for the customer value app
    ax.scatter(app_features_values[feature_selected_1], app_features_values[feature_selected_2], color='black', label='Position du client', marker='x', s=100)

    # Customize the plot
    ax.set_xlabel(feature_selected_1)
    ax.set_ylabel(feature_selected_2)
    ax.set_title('Nuage de points des crédits accordés et refusés en fonction des 2 variables sélectionnées')
    ax.legend()

    return fig

graph1 = graph_two_features(feature_selected_1,feature_selected_2)
st.pyplot(graph1)

# Charger l'image de feature importance globale
feature_importance = Image.open('feature_importance_globale.png')

# Créer une case à cocher
show_image = st.checkbox("Afficher l'importance globale des variables")

# Afficher l'image uniquement si la case à cocher est cochée
if show_image:
    st.image(feature_importance, caption="Importance globale des variables", use_column_width=True)

