import os
import sys
import json
import numpy as np
import pandas as pd
from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import CustomException

def extract_and_save_top_features(
    preprocessor_path="artifacts/preprocessor.pkl",
    model_path="artifacts/model.pkl",
    train_csv_path="artifacts/transformed_train.csv",
    output_json_path="artifacts/top_features.json",
    top_n=10
):
    try:
        logging.info("Loading preprocessor and model objects.")
        preprocessor = MainUtils.load_object(preprocessor_path)
        model = MainUtils.load_object(model_path)

        if not os.path.exists(train_csv_path):
            alt_path = "artifacts/train_processed.csv"
            if os.path.exists(alt_path):
                train_csv_path = alt_path

        # Extract feature names from preprocessor or csv header
        if isinstance(preprocessor, dict) and 'numerical_columns' in preprocessor and 'categorical_pipeline' in preprocessor:
            cat_feature_names = preprocessor['categorical_pipeline'].named_steps['onehot'].get_feature_names_out(preprocessor['categorical_columns'])
            feature_names = list(preprocessor['numerical_columns']) + list(cat_feature_names)
        else:
            feature_names = [col for col in pd.read_csv(train_csv_path, nrows=0).columns if col != "TARGET"]

        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in top_indices]

        train_df = pd.read_csv(train_csv_path, usecols=top_features)
        feature_means = {k: float(v) for k, v in train_df[top_features].mean().to_dict().items()}

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump({"top_features": top_features, "feature_means": feature_means}, f, indent=4)

        logging.info(f"Top Features and Mean saved to: {output_json_path}")
        return top_features, feature_means

    except Exception as e:
        logging.error(f"Error occurred while extracting and saving top features: {str(e)}")
        raise CustomException(e, sys)