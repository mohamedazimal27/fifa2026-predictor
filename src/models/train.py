import os
import pickle
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report
from src.data_pipeline.data_loader import merge_elo_and_results
from src.features import build_features

def main():
    results_path = "data/results.csv"
    canonical_path = "data/canonical_teams.json"
    elo_dir = "data/elo"
    curated_path = "data/curated_teams.json"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    print("Loading and merging Jürisoo + Elo data...")
    df_merged = merge_elo_and_results(results_path, canonical_path, elo_dir)
    
    # We use the optimal half-life of 365 days found in our grid search
    half_life_days = 365.0
    print(f"Building features with half-life = {half_life_days} days...")
    df_feat = build_features(df_merged, curated_path=curated_path, half_life_days=half_life_days)
    
    # Splits definition
    train_mask = (df_feat['date'] >= '2000-01-01') & (df_feat['date'] <= '2015-12-31')
    val_b_mask = (df_feat['date'] >= '2016-01-01') & (df_feat['date'] <= '2017-12-31')
    test_mask = (df_feat['date'] >= '2018-01-01') & (df_feat['date'] <= '2018-12-31')
    holdout_mask = (df_feat['date'] >= '2022-01-01') & (df_feat['date'] <= '2022-12-31')
    
    feature_cols = [
        'elo_diff', 'home_elo', 'away_elo',
        'home_form', 'away_form', 'form_diff',
        'home_squad_quality', 'away_squad_quality', 'squad_quality_diff',
        'home_interim_coach', 'away_interim_coach',
        'home_advantage', 'host_advantage_home', 'host_advantage_away'
    ]
    
    # Extract subsets
    X_train = df_feat[train_mask][feature_cols]
    y_train = df_feat[train_mask]['target']
    
    X_val_b = df_feat[val_b_mask][feature_cols]
    y_val_b = df_feat[val_b_mask]['target']
    
    X_test = df_feat[test_mask][feature_cols]
    y_test = df_feat[test_mask]['target']
    
    X_holdout = df_feat[holdout_mask][feature_cols]
    y_holdout = df_feat[holdout_mask]['target']
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(X_train)} matches")
    print(f"  Val B (Calibration): {len(X_val_b)} matches")
    print(f"  Test (2018): {len(X_test)} matches")
    print(f"  Holdout (2022): {len(X_holdout)} matches")
    
    # Calculate balanced sample weights for training set
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    print("Training base XGBoost model...")
    base_model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    base_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate base model
    loss_train = log_loss(y_train, base_model.predict_proba(X_train))
    loss_val_b = log_loss(y_val_b, base_model.predict_proba(X_val_b))
    loss_test = log_loss(y_test, base_model.predict_proba(X_test))
    loss_holdout = log_loss(y_holdout, base_model.predict_proba(X_holdout))
    
    print("\n--- Base XGBoost Performance (Uncalibrated) ---")
    print(f"  Train Log Loss: {loss_train:.5f}")
    print(f"  Val B Log Loss: {loss_val_b:.5f}")
    print(f"  Test (2018) Log Loss: {loss_test:.5f}")
    print(f"  Holdout (2022) Log Loss: {loss_holdout:.5f}")
    
    # Train Calibration Model (Platt Scaling via Multiclass Logistic Regression) on Val B
    print("\nCalibrating predictions on Val B...")
    val_b_preds = base_model.predict_proba(X_val_b)
    calibrator = LogisticRegression(
        solver='lbfgs',
        C=1.0,
        random_state=42
    )
    calibrator.fit(val_b_preds, y_val_b)
    
    # Function to get calibrated predictions
    def predict_calibrated(X):
        base_preds = base_model.predict_proba(X)
        return calibrator.predict_proba(base_preds)
        
    cal_loss_train = log_loss(y_train, predict_calibrated(X_train))
    cal_loss_val_b = log_loss(y_val_b, predict_calibrated(X_val_b))
    cal_loss_test = log_loss(y_test, predict_calibrated(X_test))
    cal_loss_holdout = log_loss(y_holdout, predict_calibrated(X_holdout))
    
    print("\n--- Calibrated Model Performance ---")
    print(f"  Train Log Loss: {cal_loss_train:.5f}")
    print(f"  Val B Log Loss: {cal_loss_val_b:.5f}")
    print(f"  Test (2018) Log Loss: {cal_loss_test:.5f}")
    print(f"  Holdout (2022) Log Loss: {cal_loss_holdout:.5f}")
    
    test_preds = predict_calibrated(X_test)
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_acc = accuracy_score(y_test, test_pred_classes)
    print(f"  Test (2018) Accuracy: {test_acc:.2%}")
    
    holdout_preds = predict_calibrated(X_holdout)
    holdout_pred_classes = np.argmax(holdout_preds, axis=1)
    holdout_acc = accuracy_score(y_holdout, holdout_pred_classes)
    print(f"  Holdout (2022) Accuracy: {holdout_acc:.2%}")
    
    # Save the model artifact
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "fifa_model.pkl")
    
    model_dict = {
        "base_model": base_model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "half_life_days": half_life_days
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"\nSaved trained and calibrated model to {model_path}")
    
    # Save benchmark metrics
    metrics_path = "models/benchmark_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            "uncalibrated": {
                "train_loss": loss_train,
                "val_b_loss": loss_val_b,
                "test_loss": loss_test,
                "holdout_loss": loss_holdout
            },
            "calibrated": {
                "train_loss": cal_loss_train,
                "val_b_loss": cal_loss_val_b,
                "test_loss": cal_loss_test,
                "holdout_loss": cal_loss_holdout,
                "test_accuracy": test_acc,
                "holdout_accuracy": holdout_acc
            }
        }, f, indent=2)
    print(f"Saved benchmark metrics to {metrics_path}")

if __name__ == "__main__":
    main()
