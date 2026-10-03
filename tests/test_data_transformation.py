import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src
from src.components.data_transformation import DataTransformation

if __name__=="__main__":
    data_transformer=DataTransformation()
    print("Starting Data Transformation process...")
    train_path, test_path, preprocessor_path = data_transformer.initiate_data_transformation()
    print("Transformed training data saved at:", train_path)
    print("Transformed test data saved at:", test_path)
    print("Transformed preprocessor object saved at:", preprocessor_path)
    print("Data Transformation process completed successfully")
    

