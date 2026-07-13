import os
import uuid
import pandas as pd
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.core.loader import Dataloader
from src.core.inspector import describe_data
from src.main import run_analysis_pipeline

app=FastAPI(title="whyanalyst.ai")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

upload_dir="storage/datasets"
os.makedirs(upload_dir,exist_ok=True)

dataset_registry={} #this will be replaced by mongodb

class AnalysisRequest(BaseModel):
    dataset_id:str
    query:str
    prediction_dataset_id:Optional[str]=None

@app.post("/upload")
async def upload_dataset(file:UploadFile=File(...)):
    if not (file.filename or "").endswith('.csv'):
        raise HTTPException(status_code=400,detail="only csv file supoorted.")
    
    dataset_id= str(uuid.uuid4()) #generating unique id
    file_path=os.path.join(upload_dir,f"{dataset_id}.csv")

    with open(file_path,"wb") as buffer:
        buffer.write(await file.read())
    
    try:
        loader=Dataloader(file_path)
        df=loader.load_data()
        if df is None:
            raise ValueError("empty or invalid csv")
        metadata=describe_data(df)
        dataset_registry[dataset_id]={
            "file_path": file_path,
            "metadata": metadata
        }

        return {
            "status":"success",
            "dataset_id": dataset_id,
            "filename": file.filename,
            "row_count":len(df)
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500,detail=f"failed to process dataset:{str(e)}")

@app.post("/analyze")
async def analyze_dataset(request: AnalysisRequest):
    #retrieving core dataset from memory
    dataset_info=dataset_registry.get(request.dataset_id)
    if not dataset_info:
        raise HTTPException(status_code=404,detail="primary dataset id not found. please upload again.")
    
    #checking if secondary dataset was passed for prediction
    new_df=None
    if request.prediction_dataset_id:
        pred_dataset_info=dataset_registry.get(request.prediction_dataset_id)
        if not pred_dataset_info:
            raise HTTPException(status_code=404, detail="Prediction dataset ID not found. Please upload it first.")
        #loading secondary dataset for prediction
        try:
            new_df=pd.read_csv(pred_dataset_info["file_path"])
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"Failed to load prediction dataset: {str(e)}")
    try:
        df=pd.read_csv(dataset_info["file_path"])
        metadata=dataset_info["metadata"]
        # main engine
        analysis_result=run_analysis_pipeline(df,metadata,request.query,new_df=new_df)
        return{"status": "success","data": analysis_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")



