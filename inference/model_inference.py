import os
import joblib
import pandas as pd
from inference.feature_extractor import feature_extractor

model_loc = 'model/'
model_name = 'decisiontree_default_setup.joblib'


class ModelInferenceService:
    """Service class for loading and predicting using the model."""

    model = None  # Placeholder for the loaded model

    @classmethod
    def load_model(cls):
        """Load the model from the specified location."""
        if cls.model is None:
            model_path = os.path.join(model_loc, model_name)
            with open(model_path, 'rb') as inp:
                cls.model = joblib.load(inp)
        return cls.model

    @classmethod
    def predict(cls, features):
        """Predict using the loaded model.

        :param features: Feature data_old for prediction.
        :return: Predicted label.
        """
        model = cls.load_model()
        return model.predict(features)


if __name__ == '__main__':
    # Define the input data_old
    input_data = {
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
input_features = feature_extractor(input_data)
model = ModelInferenceService()
model.load_model()
features_df = pd.DataFrame([input_features])
response = model.predict(features_df)
print(response)
