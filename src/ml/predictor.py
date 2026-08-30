from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import re 
import pandas as pd 

class Predictor:
    def __init__(self,preprocessor):
        self.preprocessor=preprocessor
        self.model=XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            eval_metric="mae",
            early_stopping_rounds=5,
            n_jobs=-1,
            random_state=0
        )

    def _clean_feature_name(self, name: str, cat_cols: list) -> str:
        """Dynamically formats one-hot categorical splits and numeric labels."""
        # 1. Strip remaining transformer prefixes if present
        clean_name = re.sub(r'^[a-zA-Z0-9_\-]+__', '', name) 
        # 2. Match against original categorical column names to insert standard colon delimiters
        for cat in cat_cols:
            if clean_name.startswith(f"{cat}_"):
                category_val = clean_name[len(cat) + 1:]
                col_title = cat.replace("_", " ")
                return f"{col_title}: {category_val}"
        # 3. Format numeric features (convert underscores back to space titles)
        return clean_name.replace("_", " ")

    def train(self,Xtrain,ytrain,Xvalid,yvalid):
        
        # 1. Manually fit and transform the data so XGBoost can "see" the valid set
        Xtrain_proc = self.preprocessor.fit_transform(Xtrain)
        Xvalid_proc=self.preprocessor.transform(Xvalid)
        
        # 2. Train the model directly (No Pipeline needed here)
        self.model.fit(
        Xtrain_proc, 
        ytrain,
        eval_set=[(Xvalid_proc, yvalid)], 
        verbose=False)

        # 3. NOW build the Pipeline with the ALREADY FITTED components
        # This is the secret: the Pipeline will now inherit the fitted state
        self.my_model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])

        preprocessors=self.my_model.named_steps['preprocessor']
        raw_names=preprocessors.get_feature_names_out()

        # Retrieve original categorical list dynamically from ColumnTransformer
        cat_cols = []
        for name, trans, cols in preprocessors.transformers_:
            if name == 'cat':
                cat_cols = list(cols) if isinstance(cols, (list, pd.Index)) else cols.tolist()

        # Clean all names before storing
        self.feature_names = [self._clean_feature_name(col, cat_cols) for col in raw_names]

        print("🚀 XGBoost Model trained successfully!")

    def predictvalid(self,Xvalid,yvalid):
        preds=self.my_model.predict(Xvalid)
        score=mean_absolute_error(preds,yvalid)
        return preds,score 
    
    def predicts(self,Xnew):
        return self.my_model.predict(Xnew)
    
    def get_features(self):
        return self.feature_names,self.my_model
