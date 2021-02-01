import joblib
import json
import numpy as np

from azureml.core.model import Model

def init():
    global model
    # Here "automl-model" is the name of the model registered under the workspace.
    # This call will return the path to the .pkl file on the local disk.
    model_path = Model.get_model_path(model_name='automl-model')

    # Deserialize the model files back into scikit-learn models.
    model = joblib.load(model_path)

# Note you can pass in multiple rows for scoring.
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)

        # Call predict() on each model
        result = model.predict(data)

        # You can return any JSON-serializable value.
        return {"prediction": result.tolist()}
    except Exception as e:
        result = str(e)
        return result
