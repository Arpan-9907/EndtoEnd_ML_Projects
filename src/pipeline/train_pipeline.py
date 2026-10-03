import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.utils.extract_top_features import extract_and_save_top_features
from src.utils.main_utils import MainUtils


class TrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion in TrainingPipeline...")
            data_ingestion = DataIngestion()
            data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed in TrainingPipeline.")
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self):
        try:
            logging.info("Starting data transformation in TrainingPipeline...")
            data_transformation = DataTransformation()
            train_arr_path, test_arr_path, preprocessor_path = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed in TrainingPipeline.")
            return train_arr_path, test_arr_path, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr_path, test_arr_path):
        try:
            logging.info("Starting model training in TrainingPipeline...")
            train_data = np.load(train_arr_path, allow_pickle=True).item()
            test_data = np.load(test_arr_path, allow_pickle=True).item()
            x_train, y_train = train_data['x'], train_data['y']
            x_test, y_test = test_data['x'], test_data['y']

            # Select only first 1000 rows for faster training
            # x_train, y_train = x_train[:1000], y_train[:1000]
            # x_test, y_test = x_test[:1000], y_test[:1000]

            model_trainer = ModelTrainer()
            model_path = model_trainer.initiate_model_trainer(x_train, y_train, x_test, y_test)
            logging.info(f"Model training completed. Model saved at: {model_path}")
            return model_path
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline execution...")
            self.start_data_ingestion()
            train_arr_path, test_arr_path, preprocessor_path = self.start_data_transformation()
            model_path = self.start_model_training(train_arr_path, test_arr_path)
            logging.info("Extracting and saving top features...")
            extract_and_save_top_features()
            logging.info("Training pipeline execution completed successfully.")
            return model_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()