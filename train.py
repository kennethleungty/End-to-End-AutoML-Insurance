# =========================================
# H2O AutoML Training with MLflow Tracking
# Author: Kenneth Leung
# =========================================
import argparse
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML Train and MLflow Track")

    parser.add_argument('--name', '--experiment_name',
                         metavar='', 
                        #  required=True,
                         default='insurance-automl',
                         help='Name of Experiment. Default is insurance-automl', 
                         type=str
                        )

    parser.add_argument('--target', '--t',
                        metavar='', 
                        required=True,
                        help='Name of Target Column (y)', 
                        type=str
                    )

    parser.add_argument('--models', '--m',
                        metavar='', 
                        # required=True,
                        default=10,
                        help='Number of AutoML models to train. Default is 10', 
                        type=int
                    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Initiate H2O cluster
    h2o.init()

    # Initiate MLflow client
    client = MlflowClient()

    # Get parsed experiment name
    experiment_name = args.name

    # Create MLflow experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name)
    
    mlflow.set_experiment(experiment_name)

    # Print experiment details
    print(f"Name: {experiment_name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")

    # Import data directly as H2O frame (default location is data/processed)
    main_frame = h2o.import_file(path='data/processed/train.csv')

    # Set predictor and target columns
    target = args.target
    predictors = [n for n in main_frame.col_names if n != target]

    # Factorize target variable so that autoML tackles classification problem
    main_frame[target] = main_frame[target].asfactor()

    # Wrap autoML training with MLflow
    with mlflow.start_run():
        aml = H2OAutoML(
                        max_models=args.models, # Run AutoML for n base models
                        seed=42, 
                        balance_classes=True, # Target classes imbalanced, so set this as True
                        sort_metric='logloss', # Sort models by logloss (metric for multi-classification)
                        verbosity='info', # Turn on verbose info
                        exclude_algos = ['GLM', 'DRF'], # Specify algorithms to exclude
                    )
        
        aml.train(x=predictors, y=target, training_frame=main_frame)
        
        # Set metrics to log
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("mean_per_class_error", aml.leader.mean_per_class_error())
        
        # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
        mlflow.h2o.log_model(aml.leader, 
                             artifact_path="model",
    #                          registered_model_name=''
                            )
        
        model_uri = mlflow.get_artifact_uri("model")
        print(f'AutoML best model saved in {model_uri}')
        
        # Get IDs of current experiment run
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id
        
        # Save leaderboard as CSV
        lb = get_leaderboard(aml, extra_columns='ALL')
        lb_path = f'mlruns/{exp_id}/{run_id}/artifacts/model/leaderboard.csv'
        lb.as_data_frame().to_csv(lb_path, index=False) 
        print(f'AutoML Complete. Leaderboard saved in {lb_path}')


if __name__ == "__main__":
    main()