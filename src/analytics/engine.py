import pandas as pd 

class Analytics:
    def __init__(self,metadata) :
        self.metadata=metadata

    def calculation(self,df,target_col,action,group_by_col=None):
        col_info=self.metadata['columns'].get(target_col)   #defensive programming 
        if not col_info or col_info['column_type']!='numeric':
            return f"Error: column{target_col} is not a valid column "
        
        operations={
            'mean':'mean',
            'sum': 'sum',
            'max': 'max',
            'min':'min'
        }
        op=operations.get(action) #using .get() to handle crashes 
        if not op:
            return f"Error: {action} is not a valid action"

        if group_by_col:        #for questions like average sales by region
            if group_by_col not in self.metadata['columns']:
                return f"Error: {group_by_col} is not a valid column name. Check spellings or try another column"
            result=df.groupby(group_by_col)[target_col].agg(op).reset_index()    #split-apply-combine pattern 
            return result 
        else:                 # for question like average sales 
            result=df[target_col].agg(op)
            return result 
    
        
        
            
