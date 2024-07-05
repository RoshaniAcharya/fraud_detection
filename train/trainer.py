import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, \
    precision_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import joblib


class FraudDetectionModel:
    def __init__(self, features_file, output_folder):
        self.features_file = features_file
        self.output_folder = output_folder
        self.numerical_features = ['amt', 'trans_month_sin', 'trans_month_cos', 'trans_hour_sin', 'trans_hour_cos',
                                   'age', 'distance']
        self.categorical_features = ['merchant', 'category', 'gender', 'city', 'state', 'job', 'trans_num']
        self.output_list = []
        self.output_df = pd.DataFrame()

    def preprocess_data(self):
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)])
        return preprocessor

    def train_dt(self, X_train, y_train, X_test, y_test, preprocessor):
        print("----------DECISION TREE-----------")

        # Define the parameter grid for grid search
        parameters = {
            'decisiontreeclassifier__max_depth': [4, 5, 6, 7, 8],
            'decisiontreeclassifier__criterion': ['gini', 'entropy'],
            'decisiontreeclassifier__max_features': ['sqrt', 'log2'],
            'decisiontreeclassifier__min_samples_split': [2, 6, 10, 14, 18, 22]
        }

        # Create a pipeline with preprocessing, SMOTE, and the decision tree classifier
        clf = make_pipeline(
            preprocessor,
            SMOTE(sampling_strategy='minority', random_state=5),
            DecisionTreeClassifier(random_state=5)
        )

        # Initialize GridSearchCV with the pipeline and parameter grid
        cv_combined = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, refit=True,
                                   scoring=make_scorer(accuracy_score))
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

    def create_log_dict(self, classifier, result):
        row = {
            "classifier": classifier,
            "params": result["params"],
            "f1_score": result["f1_score"],
            "train_score": result["train_score"],
            "test_score": result["test_score"]
        }
        return row

    def run(self):
        df = pd.read_csv(self.features_file)

        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42,
                                                            shuffle=True)
        X_TRAIN.reset_index(drop=True, inplace=True)
        X_TEST.reset_index(drop=True, inplace=True)
        Y_TRAIN.reset_index(drop=True, inplace=True)
        Y_TEST.reset_index(drop=True, inplace=True)

        print("Distribution of y_train = {}".format(Y_TRAIN.value_counts()))
        print("Distribution of y_test = {}".format(Y_TEST.value_counts()))

        preprocessor = self.preprocess_data()

        dt_result = self.train_dt(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, preprocessor)
        dt_row = self.create_log_dict("decision_tree", dt_result)
        self.output_list.append(dt_row)

        self.output_df = self.output_df.append(self.output_list)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        with open(self.output_folder + 'results_first_exp_cv5.csv', 'a') as f:
            self.output_df.to_csv(f, header=f.tell() == 0, index=False)
            joblib.dump(dt_result["model"], self.output_folder + 'gridsearch_exp_dt_model.joblib')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decision Tree classifier on fraud detection data.")
    parser.add_argument("--features_file", type=str, required=True, help="Path to the CSV file containing features.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder.")

    args = parser.parse_args()

    fraud_detection_model = FraudDetectionModel(args.features_file, args.output_folder)
    fraud_detection_model.run()
