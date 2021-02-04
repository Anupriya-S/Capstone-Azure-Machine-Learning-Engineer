# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Absolute Magnitude": pd.Series([0.0], dtype="float64"), "Est Dia in KM(min)": pd.Series([0.0], dtype="float64"), "Epoch Date Close Approach": pd.Series([0], dtype="int64"), "Relative Velocity km per sec": pd.Series([0.0], dtype="float64"), "Miss Dist_(kilometers)": pd.Series([0.0], dtype="float64"), "Orbit ID": pd.Series([0], dtype="int64"), "Orbit Uncertainity": pd.Series([0], dtype="int64"), "Minimum Orbit Intersection": pd.Series([0.0], dtype="float64"), "Jupiter Tisserand Invariant": pd.Series([0.0], dtype="float64"), "Epoch Osculation": pd.Series([0.0], dtype="float64"), "Eccentricity": pd.Series([0.0], dtype="float64"), "Semi Major Axis": pd.Series([0.0], dtype="float64"), "Inclination": pd.Series([0.0], dtype="float64"), "Asc Node Longitude": pd.Series([0.0], dtype="float64"), "Orbital Period": pd.Series([0.0], dtype="float64"), "Perihelion Distance": pd.Series([0.0], dtype="float64"), "Perihelion Arg": pd.Series([0.0], dtype="float64"), "Aphelion Dist": pd.Series([0.0], dtype="float64"), "Perihelion Time": pd.Series([0.0], dtype="float64"), "Mean Anomaly": pd.Series([0.0], dtype="float64"), "Mean Motion": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
