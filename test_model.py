import joblib
import os

def test_model_file():
    assert os.path.exists("model.joblib"), "Model file not found."

def test_model_load():
    model = joblib.load("model.joblib")
    assert model is not None, "Failed to load model."
