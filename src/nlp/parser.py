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
        found_row=[]

        for col in self.metadata['columns'].keys():
            pattern=rf"\b{col.lower()}[a-zA-Z]*[!?]*\b"

            if re.search(pattern,query):
                found_columns.append('col')
