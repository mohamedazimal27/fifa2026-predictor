import os
import pickle
import numpy as np
import pytest
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from src.models.train import main

def test_train_and_save(tmp_path):
    # We can check that we can load the pickle and it has the right keys
    model_path = "models/fifa_model.pkl"
    
    # Run main to train and save the model
    # (Since it uses real cached data, it will run the full training)
    # Let's verify the file is created and contains the expected components.
    if os.path.exists(model_path):
        os.remove(model_path)
        
    # Run the main training loop
    main()
    
    assert os.path.exists(model_path)
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        
    assert "base_model" in model_dict
    assert "calibrator" in model_dict
    assert "feature_cols" in model_dict
    assert "half_life_days" in model_dict
    
    assert isinstance(model_dict["base_model"], XGBClassifier)
    assert isinstance(model_dict["calibrator"], LogisticRegression)
    assert isinstance(model_dict["feature_cols"], list)
    assert model_dict["half_life_days"] == 365.0
    
    # Test prediction pipeline
    dummy_features = np.random.rand(5, len(model_dict["feature_cols"]))
    base_preds = model_dict["base_model"].predict_proba(dummy_features)
    assert base_preds.shape == (5, 3)
    
    cal_preds = model_dict["calibrator"].predict_proba(base_preds)
    assert cal_preds.shape == (5, 3)
    # Check that rows sum to 1
    assert np.allclose(cal_preds.sum(axis=1), 1.0)
