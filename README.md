# End-to-End AutoML with H2O, MLflow, FastAPI, and Streamlit for Life Insurance Risk Assessment 

Link to writeup: *Coming Soon*

## Context
- Traditional machine learning (ML) model development is time-consuming, resource-intensive, and requires a high degree of technical expertise along with many lines of code. 
- This model development process has been simplified and accelerated with the advent of automated machine learning (AutoML), allowing data teams to generate performant and scalable models quickly and efficiently.
- An important thing to remember is that there are multiple components in production-ready ML systems beyond model development.
- In this project, we explore how to set up, train, and serve an ML system using the powerful capabilities of H2O AutoML, MLflow, FastAPI, and Streamlit for life insurance risk assessment.


___
## Objective
- Build a predictive ML pipeline (on Prudential Life Insurance data) to automatically classify customer risk so that insurers can make the application process more streamlined.

___
## Pipeline Components
- Data Acquisition and Preprocessing
- H2O AutoML training with MLflow tracking
- Deployment of best H2O model via FastAPI
- Streamlit user interface to post test data to FastAPI endpoint

___
## Project Files and Folders
- `/data` - Folder containing the raw data, processed data and output data (predictions JSON file)
- `/demo` - Folder containing the gif and webm of Streamlit UI demo
- `/submissions` - Folder containing the CSV files for Kaggle submission to retrieve model accuracy scores
- `/utils` - Folder containing Python scripts with helper functions
- `01_EDA_and_Data_PreProcessing.ipynb` - Notebook detailing the data acquisition, data cleaning and feature engineering steps
- `02_XGBoost_Baseline_Model.ipynb` - Notebook running the XGBoost baseline model for subsequent comparison
- `03_H2O_AutoML_with_MLflow.ipynb` - Notebook showing the full H2O AutoML training and MLflow tracking process, along with model inference to get predictions
- `main.py` - Python script for selecting best H2O model and deploying it as FastAPI endpoint
- `train.py` - Python script for the execution of H2O AutoML training with MLflow tracking
- `ui.py` - Python script for the Streamlit web app, connected with FastAPI endpoint for model inference

___
## References
- https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
- https://www.mlflow.org/docs/latest/python_api/mlflow.h2o.html
- https://setscholars.net/automl-h2o-project-a-guide-to-build-a-multi-class-classification-model-in-python-using-car-description-data/
- https://fastapi.tiangolo.com/tutorial/request-files/
- https://medium.com/analytics-vidhya/fundamentals-of-mlops-part-4-tracking-with-mlflow-deployment-with-fastapi-61614115436
- https://testdriven.io/blog/fastapi-streamlit/
- https://rihab-feki.medium.com/deploying-machine-learning-models-with-streamlit-fastapi-and-docker-bb16bbf8eb91
- https://davidefiocco.github.io/streamlit-fastapi-ml-serving/
