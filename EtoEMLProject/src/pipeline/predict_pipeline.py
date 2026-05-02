import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import sys

class DataPrediction:
    def __init__(self):
        pass
    def initiate_data_prediction(self,features):
        preprocessor_path = r'src\components\artifacts\preprocessor.pkl'
        model_path =  r'src\components\artifacts\model.pkl'
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
        processed_data = preprocessor.transform(features)
        predicted_data = model.predict(processed_data)
        return round(predicted_data[0],2)


class DataReading:
    def __init__(self,gender,race,education,lunch,course,rscore,wscore):
        self.gender=gender
        self.race=race
        self.education = education
        self.lunch = lunch
        self.course = course
        self.rscore = rscore
        self.wscore =wscore

    def create_dataframe(self):
        try:
            data_dict = {'gender':[self.gender],'race_ethnicity':[self.race],'parental_level_of_education':[self.education],'lunch':[self.lunch],'test_preparation_course':[self.course],'reading_score':[self.rscore],'writing_score':[self.wscore]}
            df= pd.DataFrame(data_dict)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
             
