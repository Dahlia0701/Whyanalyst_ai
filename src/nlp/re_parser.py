import regex as re

class Queryparser:
    def __init__(self,metadata):
        self.metadata=metadata
        self.synonym={
            "mean": ["average", "avg", "mean"],
            "sum": ["total", "sum", "combined", "add up"],
            "max": ["highest", "max", "top", "maximum"],
            "min": ["lowest", "min", "bottom", "minimum"],
            "plot": ["chart", "graph", "plot", "visualize", "show"],
            "why": ["why", "reason", "cause", "influence", "factor", "dropped", "increased","what"],
            "predict": ["predict", "forecast", "future", "what if", "expect"]
        }

    def parser(self,query):
        query=query.lower()
        found_columns=[]
        found_action=[]
        found_values={}

        for col in self.metadata['columns'].keys():
            pattern=rf"\b{col.lower()}[a-zA-Z]*[!?.$@]*\b"

            if re.search(pattern,query):
                found_columns.append(col)

        for action,words in self.synonym.items():
            pattern=r"\b("+"|".join(words)+r")\b"
            if re.search(pattern,query):
                found_action.append(action)


        for col,info in self.metadata['columns'].items(): #for xai
            if info['column_type']=='categorical' and 'unique_values' in info:
                for val in info['unique_values']:
                    #print(f"DEBUG: Checking if '{val}' is in query...")
                    if not str(val).strip() or str(val).lower() == 'none': #it was triggering ghost matching 
                        continue
                    val_pattern = rf"\b{re.escape(str(val))}\b"
                    if re.search(val_pattern, query,re.IGNORECASE):
                        found_values[col] = val
                        
                    
        return{
            "columns": found_columns,
            "actions": found_action,
            "values": found_values,
            "original_query": query
        }
