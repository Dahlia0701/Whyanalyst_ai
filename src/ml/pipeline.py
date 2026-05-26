from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd 
from sklearn.model_selection import train_test_split

class MLPipeline:
    def __init__(self,metadata):
        self.metadata=metadata

    def prepare_data(self,df,target_col):
        #This ensures that Product Type becomes Product_Type. If you don't do this, XGBoost might raise a ValueError during the .fit() stage.
        df.columns=[c.replace(' ','_').replace('(','').replace(')','') for c in df.columns] 
        y=df[target_col]
        X=df.drop([target_col],axis=1)

        Xtrain_full,Xvalid_full,ytrain,yvalid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

        categorical_column=[cname for cname in Xtrain_full.columns if Xtrain_full[cname].nunique<10 and Xtrain_full[cname].dtype=='object']

        numerical_column=[cname for cname in Xtrain_full.columns if Xtrain_full[cname].dtype in ['int64','float64']]

        numerical_transformer=SimpleImputer(strategy='median')
        categorical_transformer=Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore',sparse='False'))
        ])

        preprocessor=ColumnTransformer(transformers=[
            ('num',numerical_transformer,numerical_column),
            ('cat',categorical_transformer,categorical_column)
        ])

        my_col=categorical_column+numerical_column
        Xtrain=Xtrain_full[my_col].copy()
        Xvalid=Xvalid_full[my_col].copy()

        return Xtrain,Xvalid,ytrain,yvalid,preprocessor