import re
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
        df=df.copy()
        #This ensures that Product Type becomes Product_Type. If you don't do this, XGBoost might raise a ValueError during the .fit() stage.
        df.columns=[c.replace(' ','_').replace('(','').replace(')','') for c in df.columns]
        df.columns=[col.strip() for col in df.columns] #also used for removing extra spaces 

        target_col = target_col.replace(' ', '_').replace('(', '').replace(')', '').strip() #also cleaing the target column because of 14th line to 
        #keep everything consistent


        id_keywords = ["id", "transaction_id", "row", "index", "date", "timestamp"]
        def is_id_like(col):
                tokens=re.split(r'[_\W]+',col.lower())
                return any(kw in tokens for kw in id_keywords)
        columns_to_drop=[c for c in df.columns if is_id_like(c)]
        
        columns_to_drop = list(set(columns_to_drop))  # Remove duplicates from our drop list and safely extract features
        feature_cols = [c for c in df.columns if c not in columns_to_drop]
            
        y=df[target_col]
        X=df[feature_cols] 

        #dropping columns that are nearly corremlated with the target(leakage) 
        leaky_cols=[]
        for col in X.select_dtypes(include='number').columns:
             corr=X[col].corr(y)
             if abs(corr)>0.95:
                  leaky_cols.append(col)
        X=X.drop(columns=leaky_cols)

        Xtrain_full,Xvalid_full,ytrain,yvalid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

        categorical_column=[cname for cname in Xtrain_full.columns if Xtrain_full[cname].nunique()<10 and 
                     not pd.api.types.is_numeric_dtype(Xtrain_full[cname])]

        numerical_column=[cname for cname in Xtrain_full.columns if Xtrain_full[cname].dtype in ['int64','float64']]

        numerical_transformer=SimpleImputer(strategy='median')
        categorical_transformer=Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
        ])

        preprocessor=ColumnTransformer(transformers=[
            ('num',numerical_transformer,numerical_column),
            ('cat',categorical_transformer,categorical_column)
        ],
        verbose_feature_names_out=False
        )

        my_col=categorical_column+numerical_column
        Xtrain=Xtrain_full[my_col].copy()
        Xvalid=Xvalid_full[my_col].copy()

        return Xtrain,Xvalid,ytrain,yvalid,preprocessor