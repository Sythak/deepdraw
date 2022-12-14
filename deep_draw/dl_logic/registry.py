from colorama import Fore, Style

import time
import mlflow
from mlflow.tracking import MlflowClient

import os
import pickle
import glob
from deep_draw.dl_logic.params import LOCAL_REGISTRY_PATH, LOCAL_REGISTRY_PATH_RNN, train_model_selection

from tensorflow.keras import Model, models

from google.cloud import storage

def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    save trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(os.environ.get("MODEL_TARGET"))
    if os.environ.get("MODEL_TARGET") == "mlflow":

        # retrieve mlflow env params
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        # configure mlflow
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT"))

        with mlflow.start_run():

            # STEP 1: push parameters to mlflow
            mlflow.log_params(params)

            # STEP 2: push metrics to mlflow
            mlflow.log_metrics(metrics)

            # STEP 3: push model to mlflow
            if model:
                if train_model_selection == 'rnn':
                    mlflow.keras.log_model(keras_model=model,
                                artifact_path="model",
                                keras_module="tensorflow.keras",
                                registered_model_name=os.environ.get("MLFLOW_MODEL_NAME_RNN"))
                elif train_model_selection == 'cnn':
                    mlflow.keras.log_model(keras_model=model,
                                artifact_path="model",
                                keras_module="tensorflow.keras",
                                registered_model_name=os.environ.get("MLFLOW_MODEL_NAME"))
        print("\n✅ data saved on mlflow")
        return None

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        if train_model_selection == 'cnn':
            params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        elif train_model_selection == 'rnn':
            params_path = os.path.join(LOCAL_REGISTRY_PATH_RNN, "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        if train_model_selection == 'cnn':
            metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        elif train_model_selection == 'rnn':
            metrics_path = os.path.join(LOCAL_REGISTRY_PATH_RNN, "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        if train_model_selection == 'cnn':
            model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
        elif train_model_selection == 'rnn':
            model_path = os.path.join(LOCAL_REGISTRY_PATH_RNN, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None


def load_model(select_model, save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    if os.environ.get("MODEL_TARGET") == "mlflow":
        stage = "Production"

        print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)

        # load model from mlflow
        model = None
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        if select_model == 'rnn':
            model_uri = f"models:/{os.environ.get('MLFLOW_MODEL_NAME_RNN')}/{stage}"
            model = mlflow.keras.load_model(model_uri=model_uri)
            return model
        elif select_model == "cnn":
            model_uri = f"models:/{os.environ.get('MLFLOW_MODEL_NAME')}/{stage}"
            model = mlflow.keras.load_model(model_uri=model_uri)
            return model

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    if select_model == 'cnn':
        model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    elif select_model == 'rnn':
        model_directory = os.path.join(LOCAL_REGISTRY_PATH_RNN, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model


def get_model_version(stage="Production"):
    """
    Retrieve the version number of the latest model in the given stage
    - stages: "None", "Production", "Staging", "Archived"
    """

    if os.environ.get("MODEL_TARGET") == "mlflow":

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        if train_model_selection == 'rnn':
            mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME_RNN")

        elif train_model_selection == 'cnn':
            mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
        client = MlflowClient()
        try:
            version = client.get_latest_versions(name=mlflow_model_name, stages=[stage])

        except:
            return None
            # check whether a version of the model exists in the given stage
        if not version:
            return None

        return int(version[0].version)

    # model version not handled

    return None

def make_gcs_bucket(bucket_name: str):
    """Create a bucket in google cloud storage"""
    # Instantiates a client
    storage_client = storage.Client()

    # Creates the new bucket
    bucket = storage_client.create_bucket(bucket_name)

    print(f"Bucket {bucket.name} created.")

def upload_gcs(bucket_name: str, local_path, destination_blob_name):
    """Upload a file to the bucket"""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.client.Client(project='Deep-draw-project', credentials=None)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_path)

    print(
        f"File {local_path} uploaded to {destination_blob_name}."
    )

def download_model_mlflow(select_model):
    stage = "Production"
    model = None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # load model from mlflow
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
    if select_model == 'rnn':
        model_uri = f"models:/{os.environ.get('MLFLOW_MODEL_NAME_RNN')}/{stage}"
    elif select_model == 'cnn':
        model_uri = f"models:/{os.environ.get('MLFLOW_MODEL_NAME')}/{stage}"
    model = mlflow.keras.load_model(model_uri=model_uri)

    if model is not None:
        if select_model == 'cnn':
            model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
        elif select_model == 'rnn':
            model_path = os.path.join(LOCAL_REGISTRY_PATH_RNN, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    return model

if __name__ == '__main__':
    get_model_version()
