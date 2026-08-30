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

    def get_contributions(self, shap_values, direction="global", top_n=10):
        """Single source of truth for feature/impact pairs — used by both the chart and the narrative."""
        if direction == "global":
            abs_importance = np.abs(shap_values).mean(axis=0)
            signed_avg = shap_values.mean(axis=0)
            df = pd.DataFrame({
                'feature': self.features,
                'impact': abs_importance,
                'typical_effect': np.where(signed_avg > 0, 'increases profit', 'decreases profit')
            })
            df = df.sort_values('impact', ascending=False).head(top_n)
            
        else:
            avg_imp = shap_values.mean(axis=0)
            df = pd.DataFrame({'feature': self.features, 'impact': avg_imp})
            df = df[df['impact'] > 0] if direction == "positive" else df[df['impact'] < 0]
            df = df.sort_values('impact', ascending=(direction == "negative")).head(top_n)
        return df.to_dict('records')

    def get_local_contributions(self, shap_values, top_n=10):
        """Signed feature contributions for a single row — keeps direction, unlike global aggregation."""
        row = shap_values[0]
        df = pd.DataFrame({'feature': self.features, 'impact': row})
        df['abs_impact'] = df['impact'].abs()
        df = df.sort_values('abs_impact', ascending=False).head(top_n).drop(columns='abs_impact')
        return df.to_dict('records')

    def generate_narrative(self, contributions: list[dict],record_context: Optional[dict]=None) -> str:
        """
        takes feature/impact pairs directly and uses Gemini to
        generate a plain-language business summary. 
        """
        if not self.client:
            print("SHAP Narrative Error : Gemini client is none (check GEMINI_API_KEY) ")
            return ""

        if not contributions:
            print("Shap Narrative Warning: No contributuons passed.")
            return ""

        try:
            context_line = ""
            if record_context:
                context_line = (
                    f"\nThe actual category values for this record are {json.dumps(record_context)}. "
                    f"Note: Sales, Quantity, and Discount were not specified by the user, so they were filled with "
                    f"dataset averages — mention this explicitly if they appear as major factors, since they reflect "
                    f"a typical value rather than something specific to this query.\n"
                )
            prompt = f"""
            You are the explainability engine for whyanalyst.ai.
            Analyze these SHAP feature impact values for an executive dashboard summary:

            Feature Contributions:{json.dumps(contributions, indent=2)}
            {context_line}
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
            return "AI narrative is temporarily unavailable."
    
    def explain(self,Xraw,query_type="global"):
        shap_values,X_proc=self.get_shap_values(Xraw)
        contributions = self.get_contributions(shap_values, direction=query_type)
        if query_type=='global':
            fig = self.plot_global(shap_values)
        elif query_type=='negative':
            fig = self.plot_directional(shap_values,direction='negative')
        elif query_type=='positive':
            fig = self.plot_directional(shap_values,direction='positive')
        return fig, contributions
        
    def plot_global(self,shap_values):
        contributions = self.get_contributions(shap_values, direction="global")
        df_imp = pd.DataFrame(contributions).rename(columns={'feature': 'Features', 'impact': 'importance'})
        df_imp = df_imp.sort_values('importance', ascending=True)
        fig=px.bar(df_imp,x='importance',y='Features',orientation='h',
                   title="<b>Global insights</b> Factors drivig overall performance",
                   template='plotly_white',color='importance',color_continuous_scale='Viridis')
        return fig
    
    def plot_directional(self,shap_values,direction="positive"):
        contributions = self.get_contributions(shap_values, direction=direction)
        df_dir = pd.DataFrame(contributions).rename(columns={'feature': 'Features'})
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



        



    
    
    



