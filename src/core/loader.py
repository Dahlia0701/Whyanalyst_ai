import pandas as pd 
import numpy as np

class Dataloader:
    def __init__(self,file_path):
        self.file_path=file_path
        self.missing_values=['Missing','missing','null','n/a','N/A','?',' ','-','Null','NULL','MISSING']

    def load_data(self):
        try:
            df=pd.read_csv(self.file_path,na_values=self.missing_values)
            return df
        except UnicodeDecodeError:
            df=pd.read_csv(self.file_path,na_values=self.missing_values,encoding='Latin-1')
            return df
        except FileNotFoundError:
            return None
    
