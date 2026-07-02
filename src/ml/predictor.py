from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

class Predictor:
    def __init__(self,preprocessor):
        self.preprocessor=preprocessor
        self.model=XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            early_stopping_rounds=5,
            n_jobs=-1,
            random_state=0
        )

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
        self.feature_names=preprocessors.get_feature_names_out()
        print("🚀 XGBoost Model trained successfully!")

    def predictvalid(self,Xvalid,yvalid):
        preds=self.my_model.predict(Xvalid)
        score=mean_absolute_error(preds,yvalid)
        return preds,score 
    
    def predicts(self,Xnew):
        return self.my_model.predict(Xnew)
    
    def get_features(self):
        return self.feature_names,self.my_model
