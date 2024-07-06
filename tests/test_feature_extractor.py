import unittest
from datetime import datetime
import numpy as np
from geopy.distance import great_circle

from inference.feature_extractor import calculate_age, get_distance, categorize_city_population, feature_extractor, \
    get_time_features


# Assuming feature_extractor and all helper functions are imported




class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Sample input data_old for testing
        self.input_data = {
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

    def test_calculate_age(self):
        dob = '1988-03-09'
        expected_age = (datetime.now() - datetime.strptime(dob, '%Y-%m-%d')).days // 365
        self.assertEqual(calculate_age(dob), expected_age)

    def test_get_distance(self):
        lat = 36.0788
        long = -81.1781
        merch_lat = 36.011293
        merch_long = -82.048315
        expected_distance = great_circle((lat, long), (merch_lat, merch_long)).miles
        self.assertAlmostEqual(get_distance(lat, long, merch_lat, merch_long), expected_distance, places=5)

    def test_categorize_city_population(self):
        city_pop = 3495
        expected_category = 'Very Small'
        self.assertEqual(categorize_city_population(city_pop), expected_category)

    def test_get_time_features(self):
        trans_date_time = '2019-01-01 00:00:18'
        trans_date_trans_time = datetime.strptime(trans_date_time, '%Y-%m-%d %H:%M:%S')
        trans_month = trans_date_trans_time.month
        trans_hour = trans_date_trans_time.hour

        expected_trans_month_sin = np.sin(2 * np.pi * trans_month / 12)
        expected_trans_month_cos = np.cos(2 * np.pi * trans_month / 12)
        expected_trans_hour_sin = np.sin(2 * np.pi * trans_hour / 24)
        expected_trans_hour_cos = np.cos(2 * np.pi * trans_hour / 24)

        trans_month_sin, trans_month_cos, trans_hour_sin, trans_hour_cos = get_time_features(trans_date_time)

        self.assertAlmostEqual(trans_month_sin, expected_trans_month_sin, places=5)
        self.assertAlmostEqual(trans_month_cos, expected_trans_month_cos, places=5)
        self.assertAlmostEqual(trans_hour_sin, expected_trans_hour_sin, places=5)
        self.assertAlmostEqual(trans_hour_cos, expected_trans_hour_cos, places=5)

    def test_feature_extraction(self):
        expected_features = {
            'merchant': 'fraud_Rippin, Kub and Mann',
            'category': 'misc_net',
            'amt': 4.97,
            'gender': 'F',
            'city': 'Moravian Falls',
            'state': 'NC',
            'job': 'Psychologist, counselling',
            'trans_num': '0b242abb623afc578575680df30655b9',
            'trans_month_sin': 0,
            'trans_month_cos': 0,
            'trans_hour_sin': 0,
            'trans_hour_cos': 0,
            'age': 0,
            'distance': 0,
            'city_pop_category': 0
        }

        extracted_features = feature_extractor(self.input_data)

        # Update expected_features with actual calculated values
        expected_features['age'] = calculate_age(self.input_data['dob'])
        expected_features['distance'] = get_distance(
            self.input_data['lat'],
            self.input_data['long'],
            self.input_data['merch_lat'],
            self.input_data['merch_long']
        )
        expected_features['city_pop_category'] = categorize_city_population(self.input_data['city_pop'])
        expected_features['trans_month_sin'], expected_features['trans_month_cos'], expected_features['trans_hour_sin'], expected_features['trans_hour_cos'] = get_time_features(self.input_data['trans_date_trans_time'])

        # Compare extracted features with expected features
        for key, value in expected_features.items():
            self.assertEqual(extracted_features[key], value)


if __name__ == '__main__':
    unittest.main()
