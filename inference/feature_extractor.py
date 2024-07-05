from datetime import datetime
import pandas as pd
import numpy as np
from geopy.distance import great_circle


def calculate_age(dob: str) -> int:
    """
    Calculate the age from the date of birth.

    :param dob: Date of birth in 'YYYY-MM-DD' format.
    :return: Age in years.
    """
    dob = datetime.strptime(dob, '%Y-%m-%d')
    age = (datetime.now() - dob).days // 365
    return age


def get_distance(lat: float, long: float, merch_lat: float, merch_long: float) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth specified by latitude and longitude.

    :param lat: Latitude of the first point.
    :param long: Longitude of the first point.
    :param merch_lat: Latitude of the second point (merchant location).
    :param merch_long: Longitude of the second point (merchant location).
    :return: Distance in miles.
    """
    coords1 = (lat, long)
    coords2 = (merch_lat, merch_long)
    distance_miles = great_circle(coords1, coords2).miles
    return distance_miles


def categorize_city_population(city_pop: int) -> str:
    """
    Categorize city population into predefined bins.

    :param city_pop: Population of the city.
    :return: Category of the city population.
    """
    bins = [0, 10000, 50000, 100000, 500000, np.inf]
    labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    category = pd.cut([city_pop], bins=bins, labels=labels, right=False)[0]
    return category


def get_time_features(trans_date_time: str):
    """
    Extract cyclical time features from transaction date and time.

    :param trans_date_time: Transaction date and time in 'YYYY-MM-DD HH:MM:SS' format.
    :return: Sin and Cos values for month and hour.
    """
    trans_date_trans_time = datetime.strptime(trans_date_time, '%Y-%m-%d %H:%M:%S')
    trans_month = trans_date_trans_time.month
    trans_hour = trans_date_trans_time.hour
    trans_month_sin = np.sin(2 * np.pi * trans_month / 12)
    trans_month_cos = np.cos(2 * np.pi * trans_month / 12)
    trans_hour_sin = np.sin(2 * np.pi * trans_hour / 24)
    trans_hour_cos = np.cos(2 * np.pi * trans_hour / 24)
    return trans_month_sin, trans_month_cos, trans_hour_sin, trans_hour_cos


def feature_extractor(input_data: dict) -> dict:
    """
    Extract features from raw input data.

    :param input_data: Dictionary containing raw input data.
    :return: Dictionary of extracted features.
    """
    feature_dict = {
        'merchant': input_data['merchant'],
        'category': input_data['category'],
        'amt': input_data['amt'],
        'gender': input_data['gender'],
        'city': input_data['city'],
        'state': input_data['state'],
        'job': input_data['job'],
        'trans_num': input_data['trans_num'],
        'trans_month_sin': 0,
        'trans_month_cos': 0,
        'trans_hour_sin': 0,
        'trans_hour_cos': 0,
        'age': 0,
        'distance': 0,
        'city_pop_category': 0
    }

    feature_dict['age'] = calculate_age(input_data['dob'])
    feature_dict['distance'] = get_distance(input_data['lat'], input_data['long'], input_data['merch_lat'],
                                            input_data['merch_long'])
    feature_dict['city_pop_category'] = categorize_city_population(input_data['city_pop'])
    feature_dict['trans_month_sin'], feature_dict['trans_month_cos'], feature_dict['trans_hour_sin'], feature_dict[
        'trans_hour_cos'] = get_time_features(input_data['trans_date_trans_time'])

    return feature_dict


if __name__ == '__main__':
    input_data = {
        'trans_date_trans_time': '2019-01-01 00:00:18',
        'merchant': 'fraud_Rippin, Kub and Mann',
        'category': 'misc_net',
        'amt': 4.97,
        'gender': 'F',
        'city': 'Moravian Falls',
        'state': 'NC',
        'lat': 36.0788,
        'long': -81.1781,
        'city_pop': 3495,
        'job': 'Psychologist, counselling',
        'dob': '1988-03-09',
        'trans_num': '0b242abb623afc578575680df30655b9',
        'merch_lat': 36.011293,
        'merch_long': -82.048315
    }
    feature_extracted = feature_extractor(input_data)
    print(feature_extracted)
