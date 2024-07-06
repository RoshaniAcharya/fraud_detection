import os
import logging
from datetime import date

import joblib
from flask import Flask, jsonify, request

from feature_extractor import feature_extractor
import pandas as pd

model_loc = 'model/'
model_name = 'decisiontree_default_setup.joblib'

logging.basicConfig(level=logging.NOTSET)
inference_logger = logging.getLogger("Inference")
inference_logger.setLevel(logging.DEBUG)

# Disable DEBUG logs from watchdog.observers.inotify_buffer
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

today = str(date.today())

LOG_FOLDER = os.path.join( 'output/logs', 'inference')

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER, exist_ok=True)

# Define file handler and set formatter
file_handler = logging.FileHandler(os.path.join(LOG_FOLDER, today + '_' + 'inference.log'))
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger
inference_logger.addHandler(file_handler)

app = Flask(__name__)


class ModelInferenceService:
    """Service class for loading and predicting using the model."""

    def __init__(self, model_loc=model_loc, model_name=model_name):
        self.model_loc = model_loc
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the model from the specified location."""
        if self.model is None:
            model_path = os.path.join(self.model_loc, self.model_name)
            with open(model_path, 'rb') as inp:
                self.model = joblib.load(inp)
        return self.model

    def predict(self, features):
        """Predict using the loaded model.

        :param features: Feature data for prediction.
        :return: Predicted label and probability
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Call load_model() first.")

        return self.model.predict(features), self.model.predict_proba(features)


model = ModelInferenceService()


@app.route('/inference', methods=['GET'])
def index():
    """Endpoint for health check."""
    return jsonify({'message': 'Inference service is running'})


@app.route('/single_inference', methods=['POST'])
def single_inference():
    """Endpoint for single inference prediction.

    :return: JSON response with prediction result.
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        input_data = data.get('input_data')

        if not input_data:
            raise ValueError("No input data provided")

        # Assuming feature_extractor function is defined to extract features from input_data
        features_for_prediction = feature_extractor(input_data)
        features_df = pd.DataFrame([features_for_prediction])
        inference_logger.debug('Running inference for user_id: {}'.format(user_id))
        response, confidence = model.predict(features_df)

        return jsonify({'message': 'Inference success', 'user_id': user_id,'label': str(response[0]),
            'confidence': confidence[0].tolist()[0]})

    except Exception as e:
        print("e", e)
        inference_logger.error('Error during inference: {}'.format(e))
        return jsonify({'message': 'Inference failed: ' + str(e), 'user_id': user_id}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
