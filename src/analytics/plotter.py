import plotly.express as px
import pandas as pd 

class Plotter:
    def __init__(self) :
        self.template="plotly_white" 
    
    def create_chart(self,result_df,target_col,group_by_col,action):
        if isinstance(result_df,str):
            return None
        if not isinstance(result_df,pd.DataFrame):
            return None
         
        title=f"{action.capitalize()} of {target_col} by {group_by_col}"
        
        figure=px.bar(
            result_df,
            x=group_by_col,
            y=target_col,
            title=title,
            template=self.template,
            color=group_by_col,
            labels={target_col:f"{action.capitalize()}Value",
                    group_by_col:f"{group_by_col.capitalize()}"}
        )
        return figure 