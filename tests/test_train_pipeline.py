import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src
from src.pipeline.train_pipeline import TrainingPipeline

if __name__ == "__main__":
    print("Testing Training Pipeline...")
    pipeline = TrainingPipeline()
    model_path = pipeline.run_pipeline()
    print(f"Model saved at: {model_path}")
    print("Training pipeline test completed successfully.")
