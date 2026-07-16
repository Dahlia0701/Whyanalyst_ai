import plotly.express as px
import pandas as pd 

class Plotter:
    def __init__(self):
        self.template = "plotly_white" 
    
    def create_chart(self, result_df, target_col, group_by_col, action):
        if isinstance(result_df, str):
            return None
        if not isinstance(result_df, pd.DataFrame):
            return None
        
        # Determine the title
        title = f"{action.capitalize()} of {target_col} by {group_by_col}"
        
        # 1. Safely extract standard Python lists from the DataFrame
        # Handle cases where index contains our grouping data or reset_index wasn't applied
        if group_by_col in result_df.columns:
            x_data = result_df[group_by_col].tolist()
            color_data = x_data  # Feed standard list for colors instead of column name
        else:
            x_data = result_df.index.tolist()
            color_data = x_data

        y_data = result_df[target_col].astype(float).tolist()  # Forces standard Python float values
         
        # 2. Build the figure passing explicit list objects
        figure = px.bar(
            x=x_data,
            y=y_data,
            title=title,
            template=self.template,
            color=color_data,
            labels={
                "x": f"{group_by_col.capitalize()}",
                "y": f"{action.capitalize()}Value",
                "color": f"{group_by_col.capitalize()}"
            }
        )
        print("X DATA:", x_data)
        print("Y DATA:", y_data)
        print(result_df)
        print(result_df.dtypes)
        figure.show()
        return figure