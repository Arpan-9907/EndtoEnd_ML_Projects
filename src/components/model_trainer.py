import os
import sys
import dataclasses
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.utils.main_utils import MainUtils


@dataclasses.dataclass
class ModelTrainerConfig:
    artifacts_folder=os.path.join("artifacts")
    trained_model_path:str=os.path.join(artifacts_folder,"model.pkl")
    expected_accuracy=0.45
    model_config_file_path=os.path.join("config", "model.yaml")

class ModelTrainer:
    
    def __init__(self):
        self.config=ModelTrainerConfig()
        self.utils=MainUtils()
        self.models={
            "RandomForestClassifier":RandomForestClassifier(n_jobs=-1),
            "XGBClassifier":XGBClassifier(),
            "GradientBoostingClassifier":GradientBoostingClassifier()
        }
        self.model_param_grid=self.utils.read_yaml_file(self.config.model_config_file_path)["model_selection"]["model"]
        
    def evaluate_models(self,X_train,y_train,X_test,y_test):
        try:
            logging.info("Evaluating models...")
            report={}
            for name,model in self.models.items():
                print(f"Training base model:{name}...")
                logging.info(f"Training {name}...")
                model.fit(X_train,y_train)
                y_pred=model.predict(X_test)
                accuracy=accuracy_score(y_test,y_pred)
                report[name]=accuracy
                logging.info(f"Model: {name}, Accuracy: {accuracy}")
                print(f"Model: {name}, Accuracy: {accuracy}")
            logging.info(f"Model evaluation report: {report}")
            print(f"Model evaluation report: {report}")
            return report
        except Exception as e:
            raise CustomException(e,sys)

    def finetune_best_model(self,model_name,model,x_train,y_train):
        try:
            print(f"Starting GridSearchCV for {model_name}...")
            logging.info(f"Starting GridSearchCV for {model_name}...")
            param_grid=self.model_param_grid[model_name]["search_param_grid"]
            grid_search=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,n_jobs=-1,verbose=1)
            grid_search.fit(x_train,y_train)
            best_params=grid_search.best_params_
            print(f"Best parameters for {model_name}: {best_params}")
            logging.info(f"Best parameters for {model_name}: {best_params}")
            model.set_params(**best_params)
            return model
        except Exception as e:
            print(f"Error during GridSearchCV for {model_name}: {e}")
            logging.error(f"Error during GridSearchCV for {model_name}: {e}")
            raise CustomException(e,sys)

    def initiate_model_trainer(self,x_train,y_train,x_test,y_test):
        logging.info("Initiating model training process...")
        try:
            logging.info("Evaluating base models")
            model_report=self.evaluate_models(x_train,y_train,x_test,y_test)
            best_model_name=max(model_report,key=model_report.get)
            best_model=self.models[best_model_name]
            logging.info(f"Best model selected: {best_model_name} with accuracy {model_report[best_model_name]}")
            
            #Fine Tune the best Model
            best_model=self.finetune_best_model(best_model_name,best_model,x_train,y_train)
            best_model.fit(x_train,y_train)
            y_pred=best_model.predict(x_test)
            final_score=accuracy_score(y_test,y_pred)
            logging.info(f"Final model {best_model_name} accuracy after fine tuning: {final_score}")
            
            if final_score<self.config.expected_accuracy:
                raise CustomException(f"No model achieved the expected accuracy of {self.config.expected_accuracy}. Best model: {best_model_name} with accuracy {model_report[best_model_name]}",sys)
            
            #Save the model
            os.makedirs(os.path.dirname(self.config.trained_model_path),exist_ok=True)
            self.utils.save_object(self.config.trained_model_path,best_model)
            logging.info(f"Best model saved at: {self.config.trained_model_path}")
            return self.config.trained_model_path
        except Exception as e:
            logging.error(f"Error during model training process: {e}")
            raise CustomException(e)
            

