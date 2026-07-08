import os 
import json
import google.generativeai as genai
from pydantic import BaseModel,Field
from typing import Dict,List,Union,Literal
from dotenv import load_dotenv

load_dotenv()

#filter format 
class Filter(BaseModel):
    column: str
    operator: Literal["=", ">", "<", ">=", "<=", "!="]
    values: Dict[str,Union[str,int,float]]

#format of llm answer
class LLMResponseSchema(BaseModel):
    columns: List[str] = Field(description="List of column names from the dataset mentioned in the query.")
    actions: List[Literal['mean', 'sum', 'max', 'min', 'plot', 'why', 'predict']] = Field(description="Operations needed.")
    filters: list[Filter]
    plan: List[Literal['calculate_stats', 'visualization', 'explainable_ai', 'prediction']] = Field(description="The execution steps.")
