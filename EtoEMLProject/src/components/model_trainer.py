from dataclasses import dataclass
import os
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model
import sys
from src.utils import save_object


@dataclass
class ModelTrainerConfig():
    model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()
    
    def initiate_model_trainer(self,tranformed_train_data,transformed_test_data):
        try:
     
            logging.info('Model Training Process Begins')
            models = {
                'LinearRegressor':LinearRegression(),
                'SupportVectorRegressor':SVR(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'KNNRegressor':KNeighborsRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'CatBRegressor':CatBoostRegressor(),
                'XGBRegressor':XGBRegressor(),
                'Ridge':Ridge(),
                'Lasso':Lasso()
            }
            x_train = tranformed_train_data[:,:-1]
            y_train = tranformed_train_data[:,-1]
            x_test =  transformed_test_data[:,:-1]
            y_test = transformed_test_data[:,-1]
            report = evaluate_model(x_train,y_train,x_test,y_test,models)
            logging.info('Final R2 Score Report Generated')
            best_model_score = max(report.values())
            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]
            logging.info(f'Best Model is {best_model_name} & Score is {best_model_score}')
            best_model = models[best_model_name]
            save_object(file_path=self.modelTrainerConfig.model_path,obj=best_model)
            logging.info('Model saved in pickle file')
        except Exception as e:
            raise CustomException(e,sys)


        




            

