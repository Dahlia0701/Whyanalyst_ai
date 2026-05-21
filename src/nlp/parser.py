import regex as re

class Queryparser:
    def __init__(self,metadata):
        self.metadata=metadata
        self.synonym={
            "mean": ["average", "avg", "mean"],
            "sum": ["total", "sum", "combined", "add up"],
            "max": ["highest", "max", "top", "maximum"],
            "min": ["lowest", "min", "bottom", "minimum"],
            "plot": ["chart", "graph", "plot", "visualize", "show"]
        }

    def parser(self,query):
        query=query.lower()
        found_columns=[]
        found_action=[]

        for col in self.metadata['columns'].keys():
            pattern=rf"\b{col.lower()}[a-zA-Z]*[!?.$@]*\b"

            if re.search(pattern,query):
                found_columns.append('col')

        for action,words in self.synonym.items():
            pattern=r"\b("+"|".join(words)+r")\b"
            if re.search(pattern,query):
                found_action.append(action)
        return{
            "columns": found_columns,
            "action": found_action,
            "original_query": query
        }
