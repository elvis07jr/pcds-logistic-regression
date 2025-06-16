import os
import pandas as pd
import subprocess
import pytest

# Define the script to be tested
SCRIPT_NAME = "generate_dataset.py"
OUTPUT_CSV_NAME = "PCDS_Diagnosis.csv"

@pytest.fixture(scope="module")
def run_script_and_load_data():
    # Run the script to generate the dataset
    subprocess.run(["python", SCRIPT_NAME], check=True)
    # Load the generated data
    df = pd.read_csv(OUTPUT_CSV_NAME)
    yield df
    # Cleanup: remove the CSV file after tests are done
    os.remove(OUTPUT_CSV_NAME)

def test_csv_created(run_script_and_load_data):
    # This test implicitly passes if run_script_and_load_data fixture runs,
    # as it would fail if PCDS_Diagnosis.csv is not created by the script.
    # For explicit check, we can assert os.path.exists, but fixture already handles it.
    assert os.path.exists(OUTPUT_CSV_NAME)

def test_csv_headers(run_script_and_load_data):
    df = run_script_and_load_data
    expected_headers = [
        "capillary_refill_time",
        "oxygen_saturation",
        "heart_rate",
        "age",
        "has_pcds",
    ]
    assert list(df.columns) == expected_headers

def test_has_pcds_column_values(run_script_and_load_data):
    df = run_script_and_load_data
    assert df["has_pcds"].isin([0, 1]).all()

def test_age_range(run_script_and_load_data):
    df = run_script_and_load_data
    assert df["age"].between(0, 120).all() # Assuming age should be within 0-120

def test_oxygen_saturation_range(run_script_and_load_data):
    df = run_script_and_load_data
    assert df["oxygen_saturation"].between(0, 100).all()
