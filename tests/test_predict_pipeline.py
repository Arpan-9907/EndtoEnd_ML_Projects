import sys
import os
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src
from src.pipeline.predict_pipeline import PredictPipeline

if __name__=="__main__":
    print("Testing Predict Pipeline(batch prediction)....")
    pipeline=PredictPipeline()
    output_path=pipeline.predict_from_csv('notebooks/data/application_test.csv')
    print(f"Prediction file treated at:{output_path}")
    print("Predict pipeline test completed successfully.")
    