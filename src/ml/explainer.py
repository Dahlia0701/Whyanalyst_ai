import shap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import cast

class Explainer:
    def __init__(self,my_model,feature_names):
        self.model=my_model.named_steps['model']
        self.preprocessor=my_model.named_steps['preprocessor']
        self.features=feature_names
        self.explainer=shap.TreeExplainer(self.model)

    def get_shap_values(self,Xraw):   #Xraw is same Xtrain of pipeline.py
        X_proc=self.preprocessor.transform(Xraw)
        shap_values=self.explainer.shap_values(X_proc)
        return shap_values,X_proc
    
    def explain(self,Xraw,query_type="global"):
        shap_values,X_proc=self.get_shap_values(Xraw)
        if query_type=='global':
            return self.plot_global(shap_values)
        elif query_type=='negative':
            return self.plot_directional(shap_values,direction='negative')
        elif query_type=='positive':
            return self.plot_directional(shap_values,direction='positive')
        
    def plot_global(self,shap_values):
        importances=np.abs(shap_values).mean(axis=0) #SHAP internally works heavily with NumPy
        df_imp=pd.DataFrame ({
            'Features': self.features,
            'importance': importances
        })
        df_imp=df_imp.sort_values('importance',ascending=True).tail(10) # take top 10 most important features 
        fig=px.bar(df_imp,x='importance',y='feature',orientation='h',
                   title="<b>Global insights</b> Factors drivig overall performance",
                   template='plotly_white',color='importance',color_continuous_scale='Viridis')
        return fig
    
    def plot_directional(self,shap_values,direction="positive"):
        avg_imp=shap_values.mean(axis=0)
        df_dir=pd.DataFrame({
            'Features': self.features,
            'impact':avg_imp
        })
        if direction=="negative":
            df_dir = cast(pd.DataFrame,df_dir[df_dir['impact'] < 0].copy())
            df_dir=df_dir.copy()
            if df_dir.empty:
                print("⚠️ No negative impacts found in this dataset.")
                return None
            else:
                df_dir=df_dir.sort_values(by="impact",ascending=True)
                title="<b>red_flags</b> Top negative Influences"
                color="Reds_r"
        

            
        else:



    
    
    



