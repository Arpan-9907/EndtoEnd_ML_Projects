import sys
import os
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    
    # Load the processed data saved by DataTransformation.
    train_data=np.load("artifacts/train.npy", allow_pickle=True).item()
    test_data=np.load("artifacts/test.npy", allow_pickle=True).item()
    x_train,y_train=train_data['x'],train_data['y']
    x_test,y_test=test_data['x'],test_data['y']
    x_train,y_train=x_train[:1000],y_train[:1000]  # Use only the first 1000 samples for training
    x_test,y_test=x_test[:200],y_test[:200]  # Use only the first 200 samples for testing
    
    trainer=ModelTrainer()
    model_path=trainer.initiate_model_trainer(x_train,y_train,x_test,y_test)
    print(f"Model saved at: {model_path}")