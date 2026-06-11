import os
import pickle
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import ks_2samp
from src.data_pipeline.data_loader import merge_elo_and_results
from src.features import build_features

def compute_ece(y_true, y_prob, n_bins=10):
    preds = np.argmax(y_prob, axis=1)
    confs = np.max(y_prob, axis=1)
    accs = (preds == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs >= bin_lower) & (confs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accs[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)

def compute_brier(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))

def get_reliability_curve(y_true, y_prob, n_bins=10):
    preds = np.argmax(y_prob, axis=1)
    confs = np.max(y_prob, axis=1)
    accs = (preds == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs >= bin_lower) & (confs < bin_upper)
        if np.sum(in_bin) > 0:
            bin_accs.append(float(np.mean(accs[in_bin])))
            bin_confs.append(float(np.mean(confs[in_bin])))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))
            bin_counts.append(0)
    return {
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts
    }

def main():
    results_path = "data/results.csv"
    canonical_path = "data/canonical_teams.json"
    elo_dir = "data/elo"
    curated_path = "data/curated_teams.json"
    squads_path = "data/squads.csv"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    print("Loading and merging Jürisoo + Elo data...")
    df_merged = merge_elo_and_results(results_path, canonical_path, elo_dir)
    
    half_life_days = 365.0
    print(f"Building features with half-life = {half_life_days} days...")
    df_feat = build_features(df_merged, curated_path=curated_path, half_life_days=half_life_days, squads_path=squads_path)
    
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
        'home_advantage', 'host_advantage_home', 'host_advantage_away',
        'home_cohesion', 'away_cohesion', 'cohesion_diff',
        'home_peak_age_ratio', 'away_peak_age_ratio', 'peak_age_ratio_diff',
        'home_veteran_ratio', 'away_veteran_ratio', 'veteran_ratio_diff',
        'home_youth_ratio', 'away_youth_ratio', 'youth_ratio_diff',
        'home_tournament_experience', 'away_tournament_experience', 'tournament_experience_diff',
        'home_momentum', 'away_momentum', 'momentum_diff'
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
    
    # Evaluate base model on Test (2018) and Holdout (2022)
    test_base_preds = base_model.predict_proba(X_test)
    holdout_base_preds = base_model.predict_proba(X_holdout)
    
    loss_train = log_loss(y_train, base_model.predict_proba(X_train))
    loss_val_b = log_loss(y_val_b, base_model.predict_proba(X_val_b))
    loss_test = log_loss(y_test, test_base_preds)
    loss_holdout = log_loss(y_holdout, holdout_base_preds)
    
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
        
    test_cal_preds = predict_calibrated(X_test)
    holdout_cal_preds = predict_calibrated(X_holdout)
    
    cal_loss_train = log_loss(y_train, predict_calibrated(X_train))
    cal_loss_val_b = log_loss(y_val_b, predict_calibrated(X_val_b))
    cal_loss_test = log_loss(y_test, test_cal_preds)
    cal_loss_holdout = log_loss(y_holdout, holdout_cal_preds)
    
    print("\n--- Calibrated Model Performance ---")
    print(f"  Train Log Loss: {cal_loss_train:.5f}")
    print(f"  Val B Log Loss: {cal_loss_val_b:.5f}")
    print(f"  Test (2018) Log Loss: {cal_loss_test:.5f}")
    print(f"  Holdout (2022) Log Loss: {cal_loss_holdout:.5f}")
    
    test_acc = accuracy_score(y_test, np.argmax(test_cal_preds, axis=1))
    holdout_acc = accuracy_score(y_holdout, np.argmax(holdout_cal_preds, axis=1))
    print(f"  Test (2018) Accuracy: {test_acc:.2%}")
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
    
    # Save benchmark metrics and detailed calibration data for Streamlit
    calibration_metrics = {
        "test_2018": {
            "before": {
                "log_loss": float(loss_test),
                "brier": compute_brier(y_test, test_base_preds),
                "ece": compute_ece(y_test, test_base_preds),
                "reliability": get_reliability_curve(y_test, test_base_preds)
            },
            "after": {
                "log_loss": float(cal_loss_test),
                "brier": compute_brier(y_test, test_cal_preds),
                "ece": compute_ece(y_test, test_cal_preds),
                "reliability": get_reliability_curve(y_test, test_cal_preds)
            }
        },
        "holdout_2022": {
            "before": {
                "log_loss": float(loss_holdout),
                "brier": compute_brier(y_holdout, holdout_base_preds),
                "ece": compute_ece(y_holdout, holdout_base_preds),
                "reliability": get_reliability_curve(y_holdout, holdout_base_preds)
            },
            "after": {
                "log_loss": float(cal_loss_holdout),
                "brier": compute_brier(y_holdout, holdout_cal_preds),
                "ece": compute_ece(y_holdout, holdout_cal_preds),
                "reliability": get_reliability_curve(y_holdout, holdout_cal_preds)
            }
        }
    }
    
    cal_metrics_path = os.path.join(models_dir, "calibration_metrics.json")
    with open(cal_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(calibration_metrics, f, indent=2)
    print(f"Saved calibration metrics to {cal_metrics_path}")
    
    # Save standard benchmark metrics file for backward compatibility
    metrics_path = os.path.join(models_dir, "benchmark_metrics.json")
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
    
    # Monitor feature drift
    print("\nComputing feature drift report (Train 2000-2015 vs Holdout 2022)...")
    drift_report = {}
    df_train = df_feat[train_mask]
    df_holdout = df_feat[holdout_mask]
    for col in feature_cols:
        val_train = df_train[col].dropna().values
        val_holdout = df_holdout[col].dropna().values
        if len(val_train) > 0 and len(val_holdout) > 0:
            stat, p_val = ks_2samp(val_train, val_holdout)
            drift_report[col] = {
                "train_mean": float(np.mean(val_train)),
                "holdout_mean": float(np.mean(val_holdout)),
                "train_std": float(np.std(val_train)),
                "holdout_std": float(np.std(val_holdout)),
                "ks_stat": float(stat),
                "p_value": float(p_val),
                "drift_detected": bool(p_val < 0.05)
            }
    
    drift_report_path = os.path.join(models_dir, "feature_drift_report.json")
    with open(drift_report_path, 'w', encoding='utf-8') as f:
        json.dump(drift_report, f, indent=2)
    print(f"Saved feature drift report to {drift_report_path}")

if __name__ == "__main__":
    main()
