import pandas as pd
from src.core.loader import Dataloader
from src.core.inspector import describe_data
from src.nlp.parser import Queryparser
from src.nlp.intent import Intentrouter
from src.analytics.engine import Analytics
from src.analytics.plotter import Plotter
from src.ml.pipeline import MLPipeline
from src.ml.predictor import Predictor
from src.ml.explainer import Explainer

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
#user_query = "Why is the profit high for Electronics in the East?" for local xai
#user_query= "What are the main drivers of high profit overall?" for positive
user_query= "PREDICT"
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
        result = engine.calculation(df, target, action,group,filter=parsed['values'])
        print(f"Step C (Result Data):\n{result}")
    else:
        print("❌ Error: No numeric column found to perform math on.")
    
# D. Run Plotter (if plan says so and result is a table)
if 'visualization' in plan and isinstance(result, pd.DataFrame):
    fig = plotter.create_chart(result, target, group, action)
    if fig:
        print("Step D: Chart generated successfully! Opening in browser...")
        #Sometimes Plotly tries to open in a way that your specific browser setup (or firewall) doesn't like. You can force it to use a simpler method.
        import plotly.io as pio    
        pio.renderers.default = 'browser'
        fig.show()

#E. IF WANT TO KNOW WHY OR PREDICT ANY DATA 
if 'prediction' in plan or 'explainable_ai' in plan :
    print("starting ML...")
    #checking pipeline
    pipeline=MLPipeline(metadata)
    target=input("enter the target column")
    target_col = target if target else "Sales"
    Xtrain,Xvalid,ytrain,yvalid,preprocessor=pipeline.prepare_data(df,target_col)
    print("Step E1: Data Prepared (Pipeline OK)")

    #checking predictor
    predictor=Predictor(preprocessor)
    predictor.train(Xtrain,ytrain,Xvalid,yvalid)
    feature_names,my_model=predictor.get_features()
    print("Step E2: Model Trained (Predictor OK)")

    if'prediction' in plan:
        preds,score=predictor.predictvalid(Xvalid,yvalid)
        loader=Dataloader("data/data2.csv")
        new_df = loader.load_data() # Use that CSV we made earlier
        if new_df is None:
            print("Error: file path doesnt exist")
            import sys 
            sys.exit()
        if target_col in new_df.columns: #removing target column
            Xnew=new_df.drop(columns=[target_col])
        else:
            Xnew=new_df.copy()
        Xnew=Xnew[Xtrain.columns]
        pre_val=predictor.predicts(Xnew)
        print("the mae is ",score)
        print("the prediction value is",pre_val)


    #checking explainer
    explainer=Explainer(my_model,feature_names)

    if parsed['values']:
        row_df=pd.DataFrame([parsed['values']])
        for col in Xtrain.columns: #handling missing columns 
            if col not in row_df.columns:
                row_df[col]=Xtrain[col].mean() if metadata['columns'][col]['column_type']=='numeric' else Xtrain[col].mode()[0] #categorical
                #the mode(most_common) return series instaed of one value thats why we used [0]
        fig=explainer.explain_local(row_df)
        print("Step E3: Local Explanation Generated")

    else:
        if any(w in user_query.lower() for w in ['low','hurt','decrease','negative']):
            query_type='negative'
        elif any(w in user_query.lower() for w in ['high','boost','increase','positive']):
            query_type='positive'
        else:
            query_type="global" 
        fig=explainer.explain(Xtrain,query_type=query_type)
        print(f"Step E3: {query_type.capitalize()} Explanation Generated")
    
    if fig:
        fig.show()






