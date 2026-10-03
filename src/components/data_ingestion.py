import os
import sys
import pandas as pd
import src
from src.constant import application_train_path, application_test_path
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    artifacts_folder: str = "artifacts"
    train_file_name: str = "application_train.csv"
    test_file_name: str = "application_test.csv"
    
    
    
class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started")
        try:
            os.makedirs(self.config.artifacts_folder, exist_ok=True)
            logging.info(f"Artifacts folder created at {self.config.artifacts_folder}")
            dst_path=os.path.join(self.config.artifacts_folder,self.config.train_file_name)
            logging.info(f"Copying data from {application_train_path} to {dst_path}")
            
            df=pd.read_csv(application_train_path)
            df.to_csv(dst_path, index=False)
            logging.info("Data Ingestion completed successfully")
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e,sys)
            
if __name__=="__main__":
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()
    logging.info("Data Ingestion process completed successfully")