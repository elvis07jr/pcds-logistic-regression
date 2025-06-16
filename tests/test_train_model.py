import os
import subprocess
import joblib
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

# Define the script to be tested and the model file it creates
TRAIN_SCRIPT_NAME = "train_model.py"
MODEL_FILE_NAME = "pcds_model.pkl"
DATA_GENERATION_SCRIPT_NAME = "generate_dataset.py"
CSV_FILE_NAME = "PCDS_Diagnosis.csv"

@pytest.fixture(scope="module")
def trained_model():
    # Ensure dataset exists before training
    if not os.path.exists(CSV_FILE_NAME):
        subprocess.run(["python", DATA_GENERATION_SCRIPT_NAME], check=True)

    # Run the training script
    subprocess.run(["python", TRAIN_SCRIPT_NAME], check=True)

    # Load the model
    model = joblib.load(MODEL_FILE_NAME)
    yield model

    # Cleanup: remove the model file and CSV file after tests
    if os.path.exists(MODEL_FILE_NAME):
        os.remove(MODEL_FILE_NAME)
    if os.path.exists(CSV_FILE_NAME):
        # To prevent interference with test_generate_dataset.py if tests run together
        # or if that script manages its own CSV.
        # Only remove if this specific fixture created it as a precondition.
        # For simplicity here, we'll assume it's okay or managed by the other test's fixture.
        # A more robust setup might involve unique filenames per test module.
        pass # Keep CSV for other tests or manage cleanup more carefully

def test_model_file_created(trained_model):
    # The fixture `trained_model` running implies the script ran and model should exist.
    # For explicit check:
    assert os.path.exists(MODEL_FILE_NAME)

def test_model_can_be_loaded(trained_model):
    assert trained_model is not None

def test_model_is_pipeline(trained_model):
    assert isinstance(trained_model, Pipeline)

def test_model_prediction(trained_model):
    # Create sample valid input data (adjust dtypes if necessary based on training data)
    sample_data = {
        "capillary_refill_time": [2.0],
        "oxygen_saturation": [98.0],
        "heart_rate": [70],
        "age": [30],
    }
    input_df = pd.DataFrame(sample_data)

    # Ensure column types match what the model's scaler expects (usually float for numeric features)
    for col in input_df.columns:
        if input_df[col].dtype == 'int64':
            input_df[col] = input_df[col].astype(float)

    prediction = trained_model.predict(input_df)
    assert prediction is not None
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]

    probabilities = trained_model.predict_proba(input_df)
    assert probabilities is not None
    assert probabilities.shape == (1, 2) # Probabilities for class 0 and class 1
    assert 0 <= probabilities[0][0] <= 1
    assert 0 <= probabilities[0][1] <= 1
