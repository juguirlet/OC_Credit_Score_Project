import pytest
from streamlit.testing.v1 import AppTest
from app.app import get_customers_ids, get_customer_values, get_features_selected, request_prediction, construire_jauge_score
import matplotlib.pyplot as plt
#import shap

@pytest.fixture
def streamlit_client():
    script_path = "app/app.py"

    # Initialize the app.
    app_test = AppTest.from_file(script_path, default_timeout=3)
    # Run the app.
    app_test.run()
    
    yield app_test

def test_get_customers_ids(streamlit_client):
    customers_ids = get_customers_ids()
    assert isinstance(customers_ids, list)

def test_get_customer_values(streamlit_client):
    customer_id = 100038
    customer_values = get_customer_values(customer_id)
    assert isinstance(customer_values, dict)

def test_get_features_selected(streamlit_client):
    features_selected_list = get_features_selected()
    assert isinstance(features_selected_list, list)

#def test_get_customer_shap_values(streamlit_client):
#    data_df = streamlit_client.cache(lambda: streamlit_client.head(1))()
#    shap_values_list, _, _ = get_customer_shap_values(data_df)
#    assert isinstance(shap_values_list, list)

def test_request_prediction(streamlit_client):
    api_url_calc = f'https://juguirlet.pythonanywhere.com/api/v1/predict'
    #data = streamlit_client.cache(lambda: streamlit_client.head(1))()
    data = streamlit_client.dataframe({'column_name': ['mocked_data']})
    response = request_prediction(api_url_calc, data)
    assert "prediction" in response

def test_construire_jauge_score(streamlit_client):
    score_remboursement_client = 0.7
    jauge_score = construire_jauge_score(score_remboursement_client)
    assert isinstance(jauge_score, plt.Figure)

if __name__ == "__main__":
    pytest.main(["-v", "--capture=no", "appli_test.py"])