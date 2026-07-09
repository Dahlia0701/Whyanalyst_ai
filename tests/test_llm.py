# test_llm.py
import pprint
from src.nlp.llm_parser import LLMParser

# 1. Create a dummy metadata dictionary that mimics your actual data structure
mock_metadata = {
    "columns": {
        "Profit": {"column_type": "numerical"},
        "Category": {"column_type": "categorical", "unique_values": ["Electronics", "Furniture"]},
        "Region": {"column_type": "categorical", "unique_values": ["East", "West"]}
    }
}

# 2. Define the complex conversational test string
test_query = "Why is the profit high for Electronics in the East?"

print("🏁 [TEST] Initializing your Gemini LLM Parser Engine...")

try:
    # 3. Feed the metadata contract into our custom class constructor
    parser = LLMParser(metadata=mock_metadata)
    
    print(f"📡 [TEST] Sending prompt to Google servers: '{test_query}'\n")
    
    # 4. Trigger the AI inference routine execution
    parsed_output = parser.parse_query(test_query)
    
    print("✅ [SUCCESS] Highly structured data returned back from Gemini:")
    
    # Pretty-print the resulting dictionary beautifully in your shell
    pprint.pprint(parsed_output)

except Exception as e:
    print(f"❌ [FAILURE] An error occurred during execution: {str(e)}")