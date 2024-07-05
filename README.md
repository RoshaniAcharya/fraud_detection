Fraud Detection Project

This project aims to develop a machine learning model to classify transactions as either fraudulent or non-fraudulent.
Given historical transaction data, including features that are relevant for detecting fraud. 
The dataset has a mix of categorical and numerical variables.
This project is divided into three parts.

 Exploratory Data Analysis:
==========================
1. Perform exploratory data analysis (EDA) to understand the data distribution, identify missing values, outliers, and any other data quality issues.
2. Preprocess the data, including handling missing values, encoding categorical variables, feature scaling, and any other necessary transformations.
 
Model Development:
==========================
1. Develop a machine learning models to predict whether a transaction is fraudulent.
2. Evaluate the models using appropriate metrics (e.g., accuracy, precision, recall, F1 score, ROC-AUC).
 
Model Deployment:
==========================
1. Develop a RESTful API using a Flask to host the selected machine learning model.
2. Have an endpoint /predict that accepts a new transaction's data (in JSON format) and returns the prediction (fraud or non-fraud) along with the prediction probability.

The project is organized into the following directories and files:



    fraud_detection/
    ├── data/                          # Directory containing the dataset
    ├── inference/                     # Directory for inference scripts
    ├── model/                         # Directory for storing trained models
    ├── notebooks/                     # Jupyter notebooks for data analysis and experiments
    ├── output/                        # Directory for logs
    ├── tests/                         # Directory for test scripts
    ├── train/                         # Directory for training scripts
    ├── Dockerfile                     # Dockerfile for containerizing the application
    ├── requirements.txt               # List of required Python packages

Setup

To set up the project, follow these steps:

    Clone the repository:

    git clone <repository-url>
    cd fraud_detection

Install the required packages:

    pip install -r requirements.txt

Run the notebooks:
    
    Open the Jupyter notebooks in the notebooks/ directory for data analysis and model training.

Usage Training the Model

To train the model, run the scripts in the train/ directory. Ensure that the data is available in the data/ directory.
    
    #TODO ADD Training script


Inference
To serve the model using FlaskAPI, run the following command:


    python inference/app.py

This will start a FlaskAPI server for making predictions on new data.
Docker

To containerize the application, use the provided Dockerfile:

    docker build -t fraud_detection .
    docker run -p 5000:5000 fraud_detection

Testing

To run unit tests, use the test.py script:

    python tests/test_feature_extractor.py
    python tests/test_flask_inference.py

Notebooks

    comparison of all models-Copy1.ipynb: Compares different machine learning models for fraud detection.
    feature_importance.ipynb: Analyzes the importance of various features in the dataset.
    fraud_eda-final.ipynb: Conducts exploratory data analysis on the dataset.