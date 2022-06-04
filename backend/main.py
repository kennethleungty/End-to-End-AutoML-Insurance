# ===========================
# Module: Backend setup (H2O, MLflowy)
# Author: Kenneth Leung
# Last Modified: 02 Jun 2022
# ===========================
# Command to execute script locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Command to run Docker image: docker run -d -p 8000:8000 <fastapi-app-name>:latest

import pandas as pd
import io
import h2o

from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

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

# Load best model (based on logloss) amongst all experiment runs
all_exps = [exp.experiment_id for exp in client.list_experiments()]
runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmin()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmin()]['experiment_id']
print(f'Loading best model: Run {run_id} of Experiment {exp_id}')
best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")

# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    test_h2o = h2o.H2OFrame(test_df)

    # Separate ID column (if any)
    id_name, X_id, X_h2o = separate_id_col(test_h2o)

    # Match test set column types with train set
    X_h2o = match_col_types(X_h2o)

    # Generate predictions with best model (output is H2O frame)
    preds = best_model.predict(X_h2o)
    
    # Apply processing if dataset has ID column
    if id_name is not None:
        preds_list = preds.as_data_frame()['predict'].tolist()
        id_list = X_id.as_data_frame()[id_name].tolist()
        preds_final = dict(zip(id_list, preds_list))
    else:
        preds_final = preds.as_data_frame()['predict'].tolist()

    # Convert predictions into JSON format
    json_compatible_item_data = jsonable_encoder(preds_final)
    return JSONResponse(content=json_compatible_item_data)

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the End to End AutoML Pipeline Project for Insurance Cross-Sell</h2>
    <p> The H2O model and FastAPI instances have been set up successfully </p>
    <p> You can view the FastAPI UI by heading to localhost:8000 </p>
    <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
    </body>
    """
    # content = """
    # <body>
    # <form action="/predict/" enctype="multipart/form-data" method="post">
    # <input name="file" type="file" multiple>
    # <input type="submit">
    # </form>
    # </body>
    # """
    return HTMLResponse(content=content)