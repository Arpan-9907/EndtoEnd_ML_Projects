import os
import sys
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.utils.main_utils import MainUtils


@dataclass
class DataTransformationConfig:
    artifacts_dir=os.path.join("artifacts")
    ingested_train_path:str=os.path.join(artifacts_dir,"application_train.csv")
    transformation_train_path:str=os.path.join(artifacts_dir,"train.npy")
    transformation_test_path:str=os.path.join(artifacts_dir,"test.npy")
    transform_train_csv_path:str=os.path.join(artifacts_dir,"transformed_train.csv")
    transform_test_csv_path:str=os.path.join(artifacts_dir,"transformed_test.csv")
    transformed_object_file_path:str=os.path.join(artifacts_dir,"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config=DataTransformationConfig()
        self.utils=MainUtils()
        
    def initiate_data_transformation(self):
        logging.info("Data Transformation has started")
        try:
            df=pd.read_csv(self.config.ingested_train_path)
            logging.info(f"Data read successfully from {self.config.ingested_train_path}")
            
            if "SK_ID_CURR" in df.columns:
                df.drop(columns=["SK_ID_CURR"], inplace=True)
                logging.info("Dropped 'SK_ID_CURR' column from the dataset")
                
            x=df.drop(columns=["TARGET"])
            y=df["TARGET"]
            logging.info("Split the dataset into features and target variable")
            
            X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
            logging.info("Data split into training and testing sets successfully")
            logging.info(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
            
            categorical_cols=X_train.select_dtypes(include=["object", "string"]).columns
            numerical_cols=X_train.select_dtypes(exclude=["object", "string"]).columns
            logging.info(f"Identified categorical columns: {list(categorical_cols)}")
            logging.info(f"Identified numerical columns: {list(numerical_cols)}")
            
            #numerical pipeline
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",RobustScaler())
            ])
            
            #Fit and transform the training data
            X_train_num=num_pipeline.fit_transform(X_train[numerical_cols])
            X_test_num=num_pipeline.transform(X_test[numerical_cols])
            logging.info("Applied numerical transformations to the training and testing sets")
            
            #Categorical pipeline
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
            ])
            X_train_cat=cat_pipeline.fit_transform(X_train[categorical_cols])
            X_test_cat=cat_pipeline.transform(X_test[categorical_cols])
            logging.info("Applied one-hot encoding to categorical features")

            #combine the numerical and categorical features
            X_train_processed=np.hstack((X_train_num,X_train_cat))
            X_test_processed=np.hstack((X_test_num,X_test_cat))
            logging.info("Combined numerical and categorical features for training and testing sets")
            
            #save the preprocessed data as npy files
            np.save(self.config.transformation_train_path, {"x": X_train_processed, "y": y_train.values})
            np.save(self.config.transformation_test_path, {"x": X_test_processed, "y": y_test.values})
            logging.info(f"Saved transformed training data to {self.config.transformation_train_path}")
            logging.info(f"Saved transformed testing data to {self.config.transformation_test_path}")
            
            #save the processed data as csv
            categorical_feature_names=cat_pipeline.named_steps["onehot"].get_feature_names_out(categorical_cols)
            all_feature_names=numerical_cols.tolist()+categorical_feature_names.tolist()
            train_df_out=pd.DataFrame(X_train_processed,columns=all_feature_names)
            test_df_out=pd.DataFrame(X_test_processed,columns=all_feature_names)
            train_df_out["TARGET"]=y_train.values
            test_df_out["TARGET"]=y_test.values
            train_df_out.to_csv(self.config.transform_train_csv_path,index=False)
            test_df_out.to_csv(self.config.transform_test_csv_path,index=False)
            logging.info(f"Saved transformed training data to {self.config.transform_train_csv_path}")
            logging.info(f"Saved transformed testing data to {self.config.transform_test_csv_path}")
            
            
            #save the preprocessor object
            preprocessor={
                'numerical_pipeline': num_pipeline,
                'categorical_pipeline': cat_pipeline,
                'numerical_columns': numerical_cols,
                'categorical_columns': X_train[categorical_cols].columns.tolist()
            }
            self.utils.save_object(self.config.transformed_object_file_path, preprocessor)
            logging.info(f"Saved preprocessor object to {self.config.transformed_object_file_path}")
            
            return(
                self.config.transformation_train_path,
                self.config.transformation_test_path,
                self.config.transformed_object_file_path
            )
            

        except Exception as e:
            logging.error("Error occurred during data transformation:%s", str(e))
            raise CustomException(e,sys)
            