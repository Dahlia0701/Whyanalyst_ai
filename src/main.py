import pandas as pd
import sys
from typing import Optional
import plotly.io as pio
from src.nlp.llm_parser import LLMParser 
from src.analytics.engine import Analytics
from src.analytics.plotter import Plotter
from src.ml.pipeline import MLPipeline
from src.ml.predictor import Predictor
from src.ml.explainer import Explainer

def run_analysis_pipeline(df: pd.DataFrame, metadata: dict, user_query: str, new_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Core execution engine. Takes the main dataframe, precomputed metadata, 
    the text query, and an optional evaluation dataframe for out-of-sample predictions.
    Returns a unified JSON-serializable structured dictionary for the frontend.
    """
    # Initialize response payload format
    response_data = {
        "query": user_query,
        "summary": "",
        "tables": [],
        "charts": [],
        "predictions": None,
        "explanation_chart": None
    }
    
    # Initialize unified LLMParser and core tools
    llm_parser = LLMParser(metadata)
    engine = Analytics(metadata)
    plotter = Plotter()

    # 1. EXECUTION FLOW: Run text query completely through your Gemini engine
    parsed = llm_parser.parse_query(user_query)
    plan = parsed.get('plan', [])
    response_data["summary"] = parsed.get('explanation', 'Analysis processed successfully.')

    # 2. ROUTER: Analytics and Plotter Engine Loop
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
            
            for f in parsed.get('filters', []):
                col = f['column']
                op = f['operator']
                val = f['value']
                
                # Dynamically execute matching conditions using pandas logic maps
                if op == "=":    filtered_df = filtered_df[filtered_df[col] == val]
                elif op == ">":   filtered_df = filtered_df[filtered_df[col] > val]
                elif op == "<":   filtered_df = filtered_df[filtered_df[col] < val]
                elif op == ">=":  filtered_df = filtered_df[filtered_df[col] >= val]
                elif op == "<=":  filtered_df = filtered_df[filtered_df[col] <= val]
                elif op == "!=":  filtered_df = filtered_df[filtered_df[col] != val]

            # MULTI-ACTION & CHART RENDERING LOOP
            for action in actions_to_run:
                # Calculate the current data table split
                result = engine.calculation(filtered_df, target, action, group, filter=None)
                
                if isinstance(result, pd.DataFrame):
                    # Format DataFrame rows into a list of row dicts for frontend rendering
                    response_data["tables"].append({
                        "action": action,
                        "data": result.reset_index().to_dict(orient="records")
                    })
                    
                    # Convert interactive plots directly to native JSON layout formats
                    if 'visualization' in plan:
                        fig = plotter.create_chart(result, target, group, action)
                        if fig:
                            response_data["charts"].append({
                                "action": action,
                                "plotly_json": pio.to_json(fig)
                            })
        else:
            response_data["summary"] += "\n[Error: No numeric column found to perform mathematical statistics on.]"

    # 3. ROUTER: MACHINE LEARNING ENGINE (Prediction & Explainable AI)
    if 'prediction' in plan or 'explainable_ai' in plan:
        pipeline = MLPipeline(metadata)
        
        # Safe fallback targets for training passes
        target_col = "Profit" if "Profit" in df.columns else "Sales"
        
        Xtrain, Xvalid, ytrain, yvalid, preprocessor = pipeline.prepare_data(df, target_col)

        predictor = Predictor(preprocessor)
        predictor.train(Xtrain, ytrain, Xvalid, yvalid)
        feature_names, my_model = predictor.get_features()

        if 'prediction' in plan:
            preds, score = predictor.predictvalid(Xvalid, yvalid)
            
            if new_df is None:
                response_data["predictions"] = {
                    "mae_score": score,
                    "predicted_values": [],
                    "message": "Model trained successfully on baseline data. Upload a secondary prediction CSV file to calculate target results."
                }
            else:
                # Dynamic matching validation layers you set up for testing targets
                if target_col in new_df.columns:
                    Xnew = new_df.drop(columns=[target_col])
                else:
                    Xnew = new_df.copy()
                    
                Xnew = Xnew[Xtrain.columns]
                pre_val = predictor.predicts(Xnew)
                
                response_data["predictions"] = {
                    "mae_score": score,
                    "predicted_values": list(pre_val)
                }

        if 'explainable_ai' in plan:
            explainer = Explainer(my_model, feature_names)

            # Collapse specific filters into single row dictionary mappings
            legacy_values_dict = {f['column']: f['value'] for f in parsed.get('filters', [])}

            if legacy_values_dict:
                row_df = pd.DataFrame([legacy_values_dict])
                for col in Xtrain.columns: 
                    if col not in row_df.columns:
                        if col in metadata['columns']:
                            row_df[col] = Xtrain[col].mean() if metadata['columns'][col]['column_type'] == 'numeric' else Xtrain[col].mode()[0]
                        else:
                            row_df[col] = Xtrain[col].mean()
                fig = explainer.explain_local(row_df)
            else:
                if any(w in user_query.lower() for w in ['low', 'hurt', 'decrease', 'negative']):
                    query_type = 'negative'
                elif any(w in user_query.lower() for w in ['high', 'boost', 'increase', 'positive']):
                    query_type = 'positive'
                else:
                    query_type = "global" 
                fig = explainer.explain(Xtrain, query_type=query_type)
            
            if fig:
                response_data["explanation_chart"] = pio.to_json(fig)
                
    return response_data