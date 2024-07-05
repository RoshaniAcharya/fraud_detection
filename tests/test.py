import requests
import json

# Define the input data
input_data = {
    'user_id': '123456',
    'input_data': {
        'Unnamed: 0': 0,
        'trans_date_trans_time': '2019-01-01 00:00:18',
        'cc_num': 2703186189652095,
        'merchant': 'fraud_Rippin, Kub and Mann',
        'category': 'misc_net',
        'amt': 4.97,
        'first': 'Jennifer',
        'last': 'Banks',
        'gender': 'F',
        'street': '561 Perry Cove',
        'city': 'Moravian Falls',
        'state': 'NC',
        'zip': 28654,
        'lat': 36.0788,
        'long': -81.1781,
        'city_pop': 3495,
        'job': 'Psychologist, counselling',
        'dob': '1988-03-09',
        'trans_num': '0b242abb623afc578575680df30655b9',
        'unix_time': 1325376018,
        'merch_lat': 36.011293,
        'merch_long': -82.048315
    }
}
# Send a POST request to the inference API
response = requests.post('http://127.0.0.1:5000/single_inference', json=input_data)

# Print the response
print(response.json())
