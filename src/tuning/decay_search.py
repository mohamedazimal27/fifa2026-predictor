import os
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from src.data_pipeline.data_loader import merge_elo_and_results
from src.features import build_features

def run_grid_search():
    results_path = "data/results.csv"
    canonical_path = "data/canonical_teams.json"
    elo_dir = "data/elo"
    curated_path = "data/curated_teams.json"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    print("Loading and merging Jürisoo + Elo data...")
    df_merged = merge_elo_and_results(results_path, canonical_path, elo_dir)
    
    # Define half-life values in days to search
    half_lives = [30, 90, 180, 270, 365, 540, 730, 1095]
    
    feature_cols = [
        'elo_diff', 'home_elo', 'away_elo',
        'home_form', 'away_form', 'form_diff',
        'home_squad_quality', 'away_squad_quality', 'squad_quality_diff',
        'home_interim_coach', 'away_interim_coach',
        'home_advantage', 'host_advantage_home', 'host_advantage_away'
    ]
    
    results = []
    
    for hl in half_lives:
        print(f"Evaluating decay half-life = {hl} days...")
        # Compute features with current half-life
        df_feat = build_features(df_merged, curated_path=curated_path, half_life_days=hl)
        
        # Split into Train (2000-2013) and Val A (2014-2015)
        # We start train at 2000-01-01
        train_df = df_feat[(df_feat['date'] >= '2000-01-01') & (df_feat['date'] <= '2013-12-31')]
        val_df = df_feat[(df_feat['date'] >= '2014-01-01') & (df_feat['date'] <= '2015-12-31')]
        
        if len(train_df) == 0 or len(val_df) == 0:
            print(f"Warning: Empty train ({len(train_df)}) or val ({len(val_df)}) set.")
            continue
            
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        
        X_val = val_df[feature_cols]
        y_val = val_df['target']
        
        # Train XGBoost
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        
        # Evaluate log loss
        preds_val = model.predict_proba(X_val)
        loss = log_loss(y_val, preds_val)
        
        print(f"Half-life {hl} days -> Val A Log Loss: {loss:.5f}")
        results.append({
            "half_life_days": hl,
            "log_loss": float(loss)
        })
        
    # Find best
    best_res = min(results, key=lambda x: x["log_loss"])
    print(f"\nBest Half-life: {best_res['half_life_days']} days with Log Loss: {best_res['log_loss']:.5f}")
    
    # Save search results
    output_path = "data/decay_search_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "best_half_life": best_res['half_life_days'],
            "all_results": results
        }, f, indent=2)
    print(f"Saved grid search results to {output_path}")

if __name__ == "__main__":
    run_grid_search()
