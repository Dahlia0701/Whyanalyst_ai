import pandas as pd 
import numpy as np

def load_data(file_path):
    missing_values=['Missing','missing','null','n/a','N/A','?',' ','-','Null','NULL','MISSING']
    try:
        df=pd.read_csv(file_path,na_values=missing_values)
        return df
    except UnicodeDecodeError:
        df=pd.read_csv(file_path,na_values=missing_values,encoding='Latin-1')
        return df
    
