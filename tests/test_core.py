import pandas as pd
import sys
from src.core.loader import Dataloader
from src.core.inspector import describe_data
from src.nlp.llm_parser import LLMParser 
from src.analytics.engine import Analytics
from src.analytics.plotter import Plotter
from src.ml.pipeline import MLPipeline
from src.ml.predictor import Predictor
from src.ml.explainer import Explainer

# 1. SETUP: Load Data & Initialize "Brain"
loader = Dataloader("data/data.csv")
df = loader.load_data() 

if df is None:
    print("Error: file path doesn't exist")
    sys.exit()

metadata = describe_data(df)

# Initialize our new unified LLMParser and core tools
llm_parser = LLMParser(metadata)
engine = Analytics(metadata)
plotter = Plotter()

# 2. TEST CASE: Complex conditional phrase
user_query = "show me the total profit by porduct category and visualize it"
print(f"\n---  Testing Query: '{user_query}' ---")

# 3. EXECUTION FLOW
# A & B. Run the text query completely through your Gemini engine
parsed = llm_parser.parse_query(user_query)
print(f"\n📡 [LLM PARSER OUTPUT]:\n{parsed}")

plan = parsed.get('plan', [])

# C & D. Run Analytics and Plotter (Combined Loop for Multiple Actions & Charts)
if 'calculate_stats' in plan:
    actions_to_run = parsed['actions'] if parsed['actions'] else ['mean']
    
    target = None
    group = None
    
    for column in parsed['columns']:
        col_info = metadata['columns'].get(column)
        if col_info:
            if col_info['column_type'] == 'numeric':
                target = column
            else:
                group = column  
    
    if target:
        # 🎯 OPERATOR ENGINE: Create a fresh copy of data to filter down
        filtered_df = df.copy()
        
        print("\nApplying dynamic query logic operations...")
        for f in parsed.get('filters', []):
            col = f['column']
            op = f['operator']
            val = f['value']
            
            print(f"   Filtering: Data Where [{col}] {op} [{val}]")
            
            # Dynamically execute matching conditions using pandas logic maps
            if op == "=":    filtered_df = filtered_df[filtered_df[col] == val]
            elif op == ">":   filtered_df = filtered_df[filtered_df[col] > val]
            elif op == "<":   filtered_df = filtered_df[filtered_df[col] < val]
            elif op == ">=":  filtered_df = filtered_df[filtered_df[col] >= val]
            elif op == "<=":  filtered_df = filtered_df[filtered_df[col] <= val]
            elif op == "!=":  filtered_df = filtered_df[filtered_df[col] != val]

        print(f"\n📊 Running Statistical calculations for: {actions_to_run}")
        
        # MULTI-ACTION & CHART RENDERING LOOP
        for action in actions_to_run:
            # 1. Calculate the current table
            result = engine.calculation(filtered_df, target, action, group, filter=None)
            print(f"\n[Action: {action.upper()}] Result Data:\n{result}")
            
            # 2. Plot the current table immediately if visualization is requested
            if 'visualization' in plan and isinstance(result, pd.DataFrame):
                print(f"Generating {action.upper()} chart...")
                fig = plotter.create_chart(result, target, group, action)
                if fig:
                    import plotly.io as pio    
                    pio.renderers.default = 'browser'
                    fig.show() # Opens a unique browser tab for each individual action!
            
    else:
        print("Error: No numeric column found to perform math on.")

# E. MACHINE LEARNING ENGINE: Prediction & Explainable AI
if 'prediction' in plan or 'explainable_ai' in plan:
    print("\n Starting ML Engine Pipeline...")
    pipeline = MLPipeline(metadata)
    
    # Safe fallback targets for training passes
    target_col = "Profit" if "Profit" in df.columns else "Sales"
    
    Xtrain, Xvalid, ytrain, yvalid, preprocessor = pipeline.prepare_data(df, target_col)
    print("Step E1: Data Prepared (Pipeline OK)")

    predictor = Predictor(preprocessor)
    predictor.train(Xtrain, ytrain, Xvalid, yvalid)
    feature_names, my_model = predictor.get_features()
    print("Step E2: Model Trained (Predictor OK)")

    if 'prediction' in plan:
        preds, score = predictor.predictvalid(Xvalid, yvalid)
        loader2 = Dataloader("data/data2.csv")
        new_df = loader2.load_data() 
        if new_df is None:
            print("Error: data2.csv path doesn't exist")
            sys.exit()
            
        if target_col in new_df.columns:
            Xnew = new_df.drop(columns=[target_col])
        else:
            Xnew = new_df.copy()
            
        Xnew = Xnew[Xtrain.columns]
        pre_val = predictor.predicts(Xnew)
        print("The MAE score is:", score)
        print("The predicted value array is:", pre_val)

    explainer = Explainer(my_model, feature_names)

    #  For single-point ML explanations, collapse filters into a single row dictionary mapping
    legacy_values_dict = {f['column']: f['value'] for f in parsed.get('filters', [])}

    if legacy_values_dict:
        row_df = pd.DataFrame([legacy_values_dict])
        for col in Xtrain.columns: 
            if col not in row_df.columns:
                row_df[col] = Xtrain[col].mean() if metadata['columns'][col]['column_type'] == 'numeric' else Xtrain[col].mode()[0]
        fig = explainer.explain_local(row_df)
        print("Step E3: Local Explanation Generated")

    else:
        if any(w in user_query.lower() for w in ['low', 'hurt', 'decrease', 'negative']):
            query_type = 'negative'
        elif any(w in user_query.lower() for w in ['high', 'boost', 'increase', 'positive']):
            query_type = 'positive'
        else:
            query_type = "global" 
        fig = explainer.explain(Xtrain, query_type=query_type)
        print(f"Step E3: {query_type.capitalize()} Explanation Generated")
    
    if fig:
        fig.show()