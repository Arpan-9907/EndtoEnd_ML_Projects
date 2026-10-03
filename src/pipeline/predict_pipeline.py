import os
import sys
import pandas as pd
import numpy as np
import json
from src.logger import logging
from src.exception import CustomException
from src.constant import target_column
from src.utils.main_utils import MainUtils

class PredictPipeline:
    def __init__(self):
        self.utils = MainUtils()
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.top_features_path = os.path.join("artifacts", "top_features.json")

    def load_artifacts(self):
        try:
            model = self.utils.load_object(self.model_path)
            preprocessor = self.utils.load_object(self.preprocessor_path)
            top_features, feature_means = [], {}
            if os.path.exists(self.top_features_path):
                with open(self.top_features_path, "r") as f:
                    features_info = json.load(f)
                top_features = features_info.get("top_features", [])
                feature_means = features_info.get("feature_means", {})
            return model, preprocessor, top_features, feature_means

        except Exception as e:
            logging.error(f"Error occurred while loading artifacts: {e}")
            raise CustomException(e, sys)

    def preprocess_input(self, input_df, preprocessor, feature_means=None):
        try:
            # Drop identifier columns as in training
            drop_cols = ['SK_ID_CURR']
            for col in drop_cols:
                if col in input_df.columns:
                    input_df = input_df.drop(columns=[col])

            numerical_cols = list(preprocessor['numerical_columns'])
            categorical_cols = list(preprocessor['categorical_columns'])
            all_features = numerical_cols + categorical_cols

            input_df = input_df.reindex(columns=all_features, fill_value=np.nan)

            # Numerical and categorical transforms using the fitted pipelines
            X_num = preprocessor['numerical_pipeline'].transform(input_df[numerical_cols])
            X_cat = preprocessor['categorical_pipeline'].transform(input_df[categorical_cols])
            X_processed = np.hstack([X_num, X_cat])
            return X_processed

        except Exception as e:
            logging.error(f"Error in preprocessing input: {e}")
            raise CustomException(e, sys)

    def predict_from_csv(self, csv_path):
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()
            input_df = pd.read_csv(csv_path)

            # If Unnamed index columns exist, drop them
            if "Unnamed: 0" in input_df.columns:
                input_df = input_df.drop(columns=["Unnamed: 0"])
            if "Unnamed:0" in input_df.columns:
                input_df = input_df.drop(columns=["Unnamed:0"])

            X_processed = self.preprocess_input(input_df.copy(), preprocessor, feature_means)
            preds = model.predict(X_processed)
            input_df = input_df.copy()
            input_df[target_column] = preds

            # Map the labels
            target_column_mapping = {0: 'bad', 1: 'good'}
            input_df[target_column] = input_df[target_column].map(target_column_mapping)

            # Print counts of each prediction
            counts = input_df[target_column].value_counts()
            print(f"Predict Counts:\n{counts.to_string()}")
            logging.info(f"Prediction Counts:\n{counts.to_string()}")

            # Save Predictions
            output_dir = os.path.join("artifacts", "predictions")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "prediction_file.csv")
            input_df.to_csv(output_path, index=False)
            logging.info(f"Predictions saved to {output_path}")
            return output_path

        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_dict(self, user_input_dict):
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()
            numerical_cols = list(preprocessor['numerical_columns'])
            categorical_cols = list(preprocessor['categorical_columns'])
            all_features = numerical_cols + categorical_cols

            # Fill missing features with np.nan so preprocessor pipeline imputes them
            full_input = {f: user_input_dict.get(f, np.nan) for f in all_features}
            input_df = pd.DataFrame([full_input])
            X_processed = self.preprocess_input(input_df, preprocessor, feature_means)
            pred = model.predict(X_processed)[0]
            return pred
        except Exception as e:
            raise CustomException(e, sys)
