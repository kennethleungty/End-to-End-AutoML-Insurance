# End-to-End AutoML with H2O, MLflow, FastAPI, and Streamlit for Insurance Cross-Sell

Link to writeup: https://towardsdatascience.com/end-to-end-automl-train-and-serve-with-h2o-mlflow-fastapi-and-streamlit-5d36eedfe606

## Overview - Business Aspect
- Cross-selling in insurance is the practice of promoting products that are complementary to the policies that existing customers already own.
- The goal of cross-selling is to create a win-win situation where customers can obtain comprehensive protection at a lower bundled cost, while insurers can boost revenue through enhanced policy conversions.
- The aim of this project is to build a predictive ML pipeline (on the Health Insurance Cross-Sell dataset) to identify health insurance customers who are interested in purchasing additional vehicle insurance, in a bid to make cross-sell campaigns more efficient and targeted.


## Overview - Technical Aspect
- Traditional machine learning (ML) model development is time-consuming, resource-intensive, and requires a high degree of technical expertise along with many lines of code. 
- This model development process has been accelerated with the advent of automated machine learning (AutoML), allowing teams to generate performant and scalable models efficiently.
- An important thing to remember is that there are multiple components in a production-ready ML system beyond model development that requires plenty of work.
- In this comprehensive guide, we explore how to set up, train, and serve an ML system using the powerful capabilities of H2O AutoML, MLflow, FastAPI, and Streamlit to build an insurance cross-sell prediction model.

___
## Objective
- Make cross-selling more efficient and targeted by building a predictive ML pipeline to identify health insurance customers interested in purchasing additional vehicle insurance.

___
## Pipeline Components
- Data Acquisition and Preprocessing
- H2O AutoML training with MLflow tracking
- Deployment of best H2O model via FastAPI
- Streamlit user interface to post test data to FastAPI endpoint

___
## UI Demo
![alt text](https://github.com/kennethleungty/End-to-End-AutoML-Insurance/blob/main/demo/streamlit-ui-2021-12-18-17-12-25.gif?raw=true)

___
## Project Files and Folders
- `/data` - Folder containing the raw data, processed data and output data (predictions JSON file)
- `/demo` - Folder containing the gif and webm of Streamlit UI demo
- `/submissions` - Folder containing the CSV files for Kaggle submission to retrieve model accuracy scores
- `/utils` - Folder containing Python scripts with helper functions
- `01_EDA_and_Data_PreProcessing.ipynb` - Notebook detailing the data acquisition, data cleaning and feature engineering steps
- `02_XGBoost_Baseline_Model.ipynb` - Notebook running the XGBoost baseline model for subsequent comparison
- `03_H2O_AutoML_with_MLflow.ipynb` - Notebook showing the full H2O AutoML training and MLflow tracking process, along with model inference to get predictions
- `train.py` - Python script for the execution of H2O AutoML training with MLflow tracking. E.g. Run in CLI with `python train.py --target 'Response'`
- `main.py` - Python script for selecting best H2O model and deploying (and serving) it as FastAPI endpoint. E.g. Run in CLI with `uvicorn main:app --host=0.0.0.0 --port=8000`
- `ui.py` - Python script for the Streamlit web app, connected with FastAPI endpoint for model inference. E.g. Run in CLI with `streamlit run ui.py`

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
