import pandas as pd
from src.core.loader import Dataloader
from src.core.inspector import describe_data
from src.nlp.parser import Queryparser
from src.nlp.intent import Intentrouter
from src.analytics.engine import Analytics
from src.analytics.plotter import Plotter

# 1. SETUP: Load Data & Initialize "Brain"
loader=Dataloader("data/data.csv")
df = loader.load_data() # Use that CSV we made earlier
if df is None:
    print("Error: file path doesnt exist")
    import sys 
    sys.exit()
# If we reach here, we know df exists
metadata = describe_data(df)
# Initialize our modules
Qparser = Queryparser(metadata)
router = Intentrouter(metadata)
engine = Analytics(metadata)
plotter = Plotter()

# 2. TEST CASE: A complex grouped query
user_query = "What is the average sales by Product_type? and plot it."
print(f"--- Testing Query: '{user_query}' ---")

# 3. EXECUTION FLOW
# A. Parse the text
parsed = Qparser.parser(user_query)
print(f"Step A (Parsed): {parsed}")

# B. Determine Plan
plan = router.intention(parsed)
print(f"Step B (Plan): {plan}")

# C. Run Analytics (if plan says so)
if 'calculate_stats' in plan:
    
    action = parsed['actions'][0]
    
    # Initialize as None to avoid "variable not defined" errors
    target = None
    group = None
    
    for column in parsed['columns']:
        # We look inside METADATA, not PARSED
        col_info = metadata['columns'].get(column)
        
        if col_info:
            if col_info['column_type'] == 'numeric':
                target = column
            else:
                # Anything not numeric (string/object) is treated as a group
                group = column  
    
    # Final check: Make sure we found at least a target column
    if target:
        result = engine.calculation(df, target, action, group)
        print(f"Step C (Result Data):\n{result}")
    else:
        print("❌ Error: No numeric column found to perform math on.")
    
# D. Run Plotter (if plan says so and result is a table)
if 'visualization' in plan and isinstance(result, pd.DataFrame):
    fig = plotter.create_chart(result, target, group, action)
    if fig:
        print("Step D: Chart generated successfully! Opening in browser...")
        fig.show()