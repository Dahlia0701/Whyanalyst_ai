import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Literal
from dotenv import load_dotenv

load_dotenv()

class FilterCondition(BaseModel):
    column: str = Field(description="The exact column name being filtered.")
    operator: Literal["=", ">", "<", ">=", "<=", "!="] = Field(description="The logical comparison operator.")
    value: Union[str, int, float] = Field(description="The single target filter value (e.g. 1989, 2000, or 'Electronics').")

# format of llm answer
class LLMResponseSchema(BaseModel):
    columns: List[str] = Field(description="List of column names from the dataset mentioned in the query.")
    actions: List[Literal['mean', 'sum', 'max', 'min', 'plot', 'why', 'predict']] = Field(description="Operations needed.")
    filters: List[FilterCondition] = Field(description="List of structural filter conditions extracted from the query.")
    plan: List[Literal['calculate_stats', 'visualization', 'explainable_ai', 'prediction']] = Field(description="The execution steps.")


class LLMParser:
    def __init__(self, metadata: dict):
        self.metadata = metadata

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL ERROR: GEMINI_API_KEY is missing from your .env file!")

        self.client = genai.Client(api_key=api_key)

    def parse_query(self, query: str) -> dict:
        available_columns = list(self.metadata.get('columns', {}).keys())

        prompt = f"""
        You are the NLP engine for whyanalyst.ai. Map this query to the dataset parameters.

        Available Dataset Columns: {available_columns}

        User Query: "{query}"
        """

        # Generate raw JSON schema mapping from our Pydantic classes
        raw_schema = LLMResponseSchema.model_json_schema()
        
        # using recursion to wipe out all 'additionalProperties'
        def strip_additional_properties(schema_dict):
            if isinstance(schema_dict, dict):
                schema_dict.pop("additionalProperties", None)
                for key, val in schema_dict.items():
                    strip_additional_properties(val)
            elif isinstance(schema_dict, list):
                for item in schema_dict:
                    strip_additional_properties(item)

        strip_additional_properties(raw_schema)

        # Sending request to Gemini
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=raw_schema 
            )
        )

        if response.text is None:
            raise ValueError("Gemini returned an empty response.")
        
        parsed_json = json.loads(response.text)
        parsed_json["original_query"] = query

        return parsed_json