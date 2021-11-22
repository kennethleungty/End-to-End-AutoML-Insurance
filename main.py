# TO DO - Add argument parsing (query parameters) - https://fastapi.tiangolo.com/tutorial/query-params/

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import pandas as pd
import os
import json
import h2o

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Create FastAPI instance
app = FastAPI()

# Initiate H2O instance and MLflow client
h2o.init()
client = MlflowClient()

# Get best model amongst all runs in all experiments
all_exps = [exp.experiment_id for exp in client.list_experiments()]
runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmin()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmin()]['experiment_id']

# Load best model (AutoML leader)
best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")

# Load dictionary of train set column types
with open('data/processed/train_col_types.json') as f:
      train_col_types = json.load(f)

# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        test_df = pd.read_csv(file.filename)
        test_df_h2o = h2o.H2OFrame(test_df)

        # Check any column is an ID
        possible_id_list = ['ID', 'Id', 'id']

        for i in possible_id_list:
            if i in test_df.columns.tolist():
                id_name = i
                X_id = test_df_h2o[:, id_name]
                X_h2o = test_df_h2o.drop(id_name)
                break
            else:
                id_name = None

        # Match test set column types with train set
        for key in train_col_types.keys():
            try:
                # If column types do not match, convert test frame col type accordingly      
                if train_col_types[key] != X_h2o.types[key]:
                    if train_col_types[key] == 'real' and X_h2o.types[key] == 'enum':
                        X_h2o[key] = X_h2o[key].ascharacter().asnumeric()
                    elif train_col_types[key] == 'real':
                        X_h2o[key] = X_h2o[key].asnumeric()
                    elif train_col_types[key] == 'int':
                        X_h2o[key] = X_h2o[key].asfactor()
                    elif train_col_types[key] == 'str':
                        X_h2o[key] = X_h2o[key].ascharacter()
            except:
                pass

        # Generate predictions with best model (output is H2O frame)
        preds = best_model.predict(X_h2o)
        
        if id_name is not None:
            preds_list = preds.as_data_frame()['predict'].tolist()
            id_list = X_id.as_data_frame()[id_name].tolist()
            preds_final = dict(zip(id_list, preds_list))
        else:
            preds_final = preds.as_data_frame()['predict'].tolist()

        json_compatible_item_data = jsonable_encoder(preds_final)
        return JSONResponse(content=json_compatible_item_data)
        
    else:
        raise ValueError(f'File path does not exist')