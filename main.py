from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import pandas as pd
import h2o

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from utils.data_processing import match_col_types, separate_id_col

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

# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        test_df = pd.read_csv(file.filename)
        test_h2o = h2o.H2OFrame(test_df)

        # Separate ID column (if any)
        id_name, X_id, X_h2o = separate_id_col(test_h2o)

        # Match test set col types with train set
        X_h2o = match_col_types(X_h2o)

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
        raise ValueError(f'Uploaded file is not a valid CSV file')