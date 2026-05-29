import pandas as pd

def describe_data(df:pd.DataFrame):
    metadata={
        'total_rows':len(df),
        'total_col':len(df.columns),
        'columns':{}
        }
    
    for col in df.columns:
        unique_val=df[col].unique()
        unique_count=len(unique_val)
        missing_val=int(df[col].isnull().sum())

        if pd.api.types.is_numeric_dtype(df[col]):  #helper function used here pd.api.types.isnumeric_dtype is not only for integer but also for floats and unsigned bits 
            col_type='numeric'
            extra_info={ 'min':float(df[col].min()) if missing_val<len(df) else None,
                        'max':float(df[col].max()) if missing_val<len(df) else None
                        }
        
        elif unique_count<(len(df)*0.05) or unique_count<=5: #low cardinality for big dataset and small_set_category for small dataset
            col_type='categorical'
            extra_info={ 'unique_values': str(val) for val in unique_val[:15] } #unique_values have top 15 unique values from the list of unique values 

        else:
            col_type='identifier/text'
            extra_info={}

        metadata['columns'][col]= {
            'column_type':col_type,
            'unique_count':unique_count,
            'null_count':missing_val,
            'sample_data': [str(x) for x in df[col].dropna().head(3).tolist()], # like for col 'age' sample data will be(top3data) ['3','7','8']
            **extra_info # dictionary unpacking , It inserts all key-value pairs from another dictionary.
            }
    return metadata

#testing metadata 
'''
if __name__ == "__main__":
    # Small self-test
    data = {
        'Sales': [100, 200, 150],
        'Region': ['North', 'South', 'North'],
        'ID': ['TX1', 'TX2', 'TX3']
    }
    test_df = pd.DataFrame(data)
    print(describe_data(test_df))
   '''     
#output for reference
''' 
metadata = {
    'total_rows': 3,
    'total_col': 3,
    'columns': {
        'Sales': {
            'column_type': 'numeric',
            'unique_count': 3,
            'null_count': 0,
            'sample_data': ['100', '200', '150'],
            'min': 100.0,
            'max': 200.0
        },
        'Region': {
            'column_type': 'categorical',
            'unique_count': 2,
            'null_count': 0,
            'sample_data': ['North', 'South', 'North'],
            'options': ['North', 'South']
        },
        'ID': {
            'column_type': 'identifier/text',
            'unique_count': 3,
            'null_count': 0,
            'sample_data': ['TX1', 'TX2', 'TX3']
        }
    }
}
'''