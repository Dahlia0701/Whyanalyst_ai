import os
import json
import shap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import cast,Optional
from google import genai
from dotenv import load_dotenv

load_dotenv()

class Explainer:
    def __init__(self,my_model,feature_names):
        self.model=my_model.named_steps['model']
        self.preprocessor=my_model.named_steps['preprocessor']
        self.features=feature_names
        self.explainer=shap.TreeExplainer(self.model)

        api_key=os.getenv("GEMINI_API_KEY")
        self.client=genai.Client(api_key=api_key) if api_key else None

    def get_shap_values(self,Xraw):   #Xraw is same Xtrain of pipeline.py
        X_proc=self.preprocessor.transform(Xraw)
        shap_values=self.explainer.shap_values(X_proc)
        return shap_values,X_proc

    def generate_narrative(self, fig) -> str:
        """
        Extracts numerical trace data directly from the Plotly figure (Bar or Waterfall)
        and uses Gemini to generate a plain-language business summary.
        """
        
        if not self.client:
            print("⚠️ SHAP Narrative Error: Gemini client is None (check GEMINI_API_KEY).")
            return ""

        if not fig or not hasattr(fig, 'data') or len(fig.data) == 0:
            print("⚠️ SHAP Narrative Error: Empty figure passed.")
            return ""

        try:
            trace = fig.data[0]
            
            # Robust extraction of x and y from any Plotly trace type
            raw_y = list(trace.y) if hasattr(trace, 'y') and trace.y is not None else []
            raw_x = list(trace.x) if hasattr(trace, 'x') and trace.x is not None else []

            feature_names = list(raw_y) if raw_y is not None else []
            shap_values = list(raw_x) if raw_x is not None else []

            # Match features and values safely
            contributions = []
            for f, v in zip(raw_y, raw_x):
                if f is not None and v is not None:
                    try:
                        contributions.append({"feature": str(f), "impact": float(v)})
                    except (ValueError, TypeError):
                        continue

            if not contributions:
                print("⚠️ SHAP Narrative Warning: No valid contribution pairs extracted from figure.")
                return ""


            prompt = f"""
            You are the explainability engine for whyanalyst.ai.
            Analyze these SHAP feature impact values for an executive dashboard summary:

            Feature Contributions:
            {json.dumps(contributions, indent=2)}

            Instructions:
            - Provide a concise 2-3 sentence executive narrative explaining what factors drove the outcome up or pulled it down.
            - Write in clear business language suitable for non-technical stakeholders.
            - Do NOT include raw SHAP scores, technical jargon (like 'waterfall', 'base value', 'log-odds'), or markdown bold formatting.
            - Keep the output as clean plain text.
            """

            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )

            return response.text.strip() if response.text else ""

        except Exception as e:
            print(f"⚠️ SHAP Narrative Generation Error: {e}")
            return ""
    
    def explain(self,Xraw,query_type="global"):
        shap_values,X_proc=self.get_shap_values(Xraw)
        if query_type=='global':
            return self.plot_global(shap_values)
        elif query_type=='negative':
            return self.plot_directional(shap_values,direction='negative')
        elif query_type=='positive':
            return self.plot_directional(shap_values,direction='positive')
        
    def plot_global(self,shap_values):
        importances=np.abs(shap_values).mean(axis=0) #SHAP internally works heavily with NumPy
        df_imp=pd.DataFrame ({
            'Features': self.features,
            'importance': importances
        })
        df_imp=df_imp.sort_values('importance',ascending=True).tail(10) # take top 10 most important features 
        fig=px.bar(df_imp,x='importance',y='Features',orientation='h',
                   title="<b>Global insights</b> Factors drivig overall performance",
                   template='plotly_white',color='importance',color_continuous_scale='Viridis')
        return fig
    
    def plot_directional(self,shap_values,direction="positive"):
        avg_imp=shap_values.mean(axis=0)
        df_dir=pd.DataFrame({
            'Features': self.features,
            'impact':avg_imp
        })
        if direction=="negative":
            df_dir = cast(pd.DataFrame,df_dir[df_dir['impact'] < 0].copy()) #cast is used to tell pylance that variable is a dataframe
            if df_dir.empty:
                print("⚠️ No negative impacts found in this dataset.")
                return None
            else:
                df_dir=df_dir.sort_values(by="impact",ascending=True)
                title="<b>red_flags</b> Top negative Influences"
                color="Reds_r" #reversed red dark to light
        else:
            df_dir = cast(pd.DataFrame,df_dir[df_dir['impact'] > 0].copy()) #cast is used to tell pylance that variable is a dataframe
            if df_dir.empty:
                print("⚠️ No positive impacts found in this dataset.")
                return None
            else:
                df_dir=df_dir.sort_values(by="impact",ascending=False)
                title="<b>Success factors</b> Top positive Influences"
                color="Greens" #light to green 
        fig=px.bar(df_dir.head(10),x='impact',y='Features',orientation='h',title=title,template='plotly_white',
                   color='impact',color_continuous_scale=color)
        print(f"DEBUG: Chart Data Shape: {df_dir.shape}")
        print("DEBUG: Attempting to open browser...")

        return fig
    
    def explain_local(self,single_row_df):
        shap_values,_=self.get_shap_values(single_row_df)
        fig=go.Figure(go.Waterfall(    #waterfall wrapping it in figure.go.figure creates a full Figure and puts that waterfall trace(chart data) inside it.
            orientation='h',
            x=shap_values[0],
            y=self.features,
            connector={'line':{'color':'rgb(63,63,63)'}},
            
        ))
        # 3. go.waterfall is an object that plot data points and wrapping it in go.figure creates fig the overall chart with axis ,colors etc
        #so the title template object lies in update_layout after putting our datapoints (waterfall) inside figure(the overall chart)
        fig.update_layout(
        title="Local Explanation: Why this specific record?",
        template="plotly_white",
        showlegend=False)

        return fig



        



    
    
    



