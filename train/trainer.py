#!/usr/bin/env python
# coding: utf-8

# ## Import Required Libraries

# In[49]:


import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, f1_score

# Decision Tree


def train_dt(X_train, y_train, X_test, y_test, preprocessor):
    print("----------DECISION TREE-----------")

    # Define the parameter grid for grid search
    parameters = {
        'decisiontreeclassifier__max_depth':  [4,5,6,7,8],
        'decisiontreeclassifier__criterion': ['gini', 'entropy'],
        'decisiontreeclassifier__max_features': ['sqrt', 'log2'],
        'decisiontreeclassifier__min_samples_split': [2,6,10,14,18,22]
    }

    # Create a pipeline with preprocessing, SMOTE, and the decision tree classifier
    clf = make_pipeline(
        preprocessor,
        SMOTE(sampling_strategy='minority', random_state=5),
        DecisionTreeClassifier(random_state=5)
    )

    # Initialize GridSearchCV with the pipeline and parameter grid
    cv_combined = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, refit=True, scoring=make_scorer(accuracy_score))
    cv_combined.fit(X_train, y_train)

    print('Best parameters: ', cv_combined.best_params_)

    # Make predictions on the test set
    y_pred = cv_combined.predict(X_test)
    
    # Calculate and print the F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f1)

    # Return a dictionary with relevant metrics and the best model
    return {
        "f1_score": f1,
        "params": cv_combined.best_params_,
        "train_score": cv_combined.best_score_,
        "test_score": accuracy_score(y_test, y_pred),
        "model": cv_combined
    }


def create_log_dict(classifier, result):
    row = {
           "classifier": classifier,
           "params": result["params"],
           "f1_score": result["f1_score"],
           "train_score": result["train_score"],
           "test_score": result["test_score"]
          }
    return row


def preprocess_data():
    numerical_features = ['amt','trans_month_sin', 'trans_month_cos', 'trans_hour_sin', 'trans_hour_cos','age','distance']
    categorical_features = ['merchant', 'category','gender','city','state', 'job','trans_num']
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
    return preprocessor


features_file = "/home/fm-pc-lt-173/Desktop/fraud_exps/final_features.csv"
output_list = []
OUTPUT_FOLDER = '/home/fm-pc-lt-173/Desktop/fraud_exps/dt_grid_search_result/'
output_df = pd.DataFrame()

df = pd.read_csv(features_file)

X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X,y,stratify=y,test_size = 0.2,random_state=42,shuffle=True)
X_TRAIN.reset_index(drop=True,inplace=True)
X_TEST.reset_index(drop=True,inplace=True)
Y_TRAIN.reset_index(drop=True,inplace=True)
Y_TEST.reset_index(drop=True,inplace=True)

print("Distribution of y_train = {}".format(Y_TRAIN.value_counts()))
print("Distribution of y_test = {}".format(Y_TEST.value_counts()))

preprocessor = preprocess_data()

dt_result = train_dt(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, preprocessor)
dt_row = create_log_dict("decision_tree", dt_result)
output_list.append(dt_row)

output_df = output_df.append(output_list)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
with open(OUTPUT_FOLDER + f'results_first_exp_cv5.csv', 'a') as f:
    output_df.to_csv(f, header=f.tell()==0, index=False)
    joblib.dump(dt_result["model"], OUTPUT_FOLDER + f'gridsearch_exp_dt_model.joblib')


