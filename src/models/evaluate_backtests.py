import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from src.data_pipeline.data_loader import merge_elo_and_results
from src.features import build_features

def get_stage_2018(date_str):
    if date_str <= '2018-06-28':
        return 'Group Stage'
    elif date_str <= '2018-07-03':
        return 'Round of 16'
    elif date_str <= '2018-07-07':
        return 'Quarter-finals'
    elif date_str <= '2018-07-11':
        return 'Semi-finals'
    elif date_str <= '2018-07-14':
        return 'Third Place Play-off'
    else:
        return 'Final'

def get_stage_2022(date_str):
    if date_str <= '2022-12-02':
        return 'Group Stage'
    elif date_str <= '2022-12-06':
        return 'Round of 16'
    elif date_str <= '2022-12-10':
        return 'Quarter-finals'
    elif date_str <= '2022-12-14':
        return 'Semi-finals'
    elif date_str <= '2022-12-17':
        return 'Third Place Play-off'
    else:
        return 'Final'

def main():
    results_path = "data/results.csv"
    canonical_path = "data/canonical_teams.json"
    elo_dir = "data/elo"
    curated_path = "data/curated_teams.json"
    squads_path = "data/squads.csv"
    model_path = "models/fifa_model.pkl"
    
    print("Loading datasets for historical backtests...")
    df_merged = merge_elo_and_results(results_path, canonical_path, elo_dir)
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        
    base_model = model_dict["base_model"]
    calibrator = model_dict["calibrator"]
    feature_cols = model_dict["feature_cols"]
    half_life_days = model_dict.get("half_life_days", 365.0)
    
    print("Engineering features...")
    df_feat = build_features(df_merged, curated_path=curated_path, half_life_days=half_life_days, squads_path=squads_path)
    
    # Filter for World Cup matches in 2018 and 2022
    wc_2018_mask = (df_feat['date'] >= '2018-01-01') & (df_feat['date'] <= '2018-12-31') & (df_feat['tournament'] == 'FIFA World Cup')
    wc_2022_mask = (df_feat['date'] >= '2022-01-01') & (df_feat['date'] <= '2022-12-31') & (df_feat['tournament'] == 'FIFA World Cup')
    
    backtest_data = {}
    
    for year, mask, stage_func in [(2018, wc_2018_mask, get_stage_2018), (2022, wc_2022_mask, get_stage_2022)]:
        df_wc = df_feat[mask].copy()
        if df_wc.empty:
            print(f"Warning: No World Cup matches found for {year} in features.")
            continue
            
        X = df_wc[feature_cols]
        y = df_wc['target'].values
        
        # Predict outcome probabilities
        base_preds = base_model.predict_proba(X)
        cal_preds = calibrator.predict_proba(base_preds)
        
        loss = log_loss(y, cal_preds, labels=[0, 1, 2])
        acc = accuracy_score(y, np.argmax(cal_preds, axis=1))
        
        matches_list = []
        for i, (_, row) in enumerate(df_wc.iterrows()):
            date_str = row['date'].strftime('%Y-%m-%d')
            actual_outcome = int(y[i]) # 0: Home Win, 1: Draw, 2: Away Win
            prob_home, prob_draw, prob_away = map(float, cal_preds[i])
            
            matches_list.append({
                "date": date_str,
                "stage": stage_func(date_str),
                "home_team": row['home_team'],
                "away_team": row['away_team'],
                "home_score": int(row['home_score']),
                "away_score": int(row['away_score']),
                "actual_outcome": actual_outcome,
                "probabilities": {
                    "home": prob_home,
                    "draw": prob_draw,
                    "away": prob_away
                }
            })
            
        # Group matches by stage for presentation
        backtest_data[str(year)] = {
            "overall_log_loss": float(loss),
            "overall_accuracy": float(acc),
            "matches": matches_list
        }
        print(f"World Cup {year}: Accuracy = {acc:.2%}, Log Loss = {loss:.4f}")
        
    os.makedirs("models", exist_ok=True)
    with open("models/historical_backtests.json", "w", encoding='utf-8') as f:
        json.dump(backtest_data, f, indent=2)
    print("Saved historical backtests to models/historical_backtests.json")

if __name__ == "__main__":
    main()
