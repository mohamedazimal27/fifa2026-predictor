import os
import json
import tempfile
import pandas as pd
import numpy as np
from src.features import build_features

def test_build_features():
    # 1. Create a dummy merged DataFrame
    data = {
        'date': pd.to_datetime(['2014-06-12', '2014-06-19', '2014-06-26']),
        'home_team': ['Brazil', 'Brazil', 'Croatia'],
        'away_team': ['Croatia', 'Mexico', 'Mexico'],
        'home_score': [3, 0, 1],
        'away_score': [1, 0, 3],
        'home_elo_before': [2000, 2010, 1850],
        'away_elo_before': [1800, 1900, 1905],
        'neutral': [False, False, True],
        'country': ['Brazil', 'Brazil', 'Brazil']
    }
    df = pd.DataFrame(data)
    
    # 2. Create a dummy curated_teams.json
    curated_data = {
        "Brazil": {
            "coaches": [
                {"name": "Luiz Felipe Scolari", "start": "2012-11-28", "end": "2014-07-14"}
            ],
            "squad_quality": [
                {"start": "2014-01-01", "end": "2014-12-31", "score": 1.20}
            ]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(curated_data, f)
        temp_path = f.name
        
    try:
        res_df = build_features(df, curated_path=temp_path, half_life_days=30, max_lookback=5)
        
        # Check lengths
        assert len(res_df) == 3
        
        # Check target
        assert res_df.iloc[0]['target'] == 0 # Win (3-1)
        assert res_df.iloc[1]['target'] == 1 # Draw (0-0)
        assert res_df.iloc[2]['target'] == 2 # Loss (1-3)
        
        # Check Elo diff
        assert res_df.iloc[0]['elo_diff'] == 200.0
        assert res_df.iloc[1]['elo_diff'] == 110.0
        
        # Check home advantage
        assert res_df.iloc[0]['home_advantage'] == 1 # Brazil at Brazil, neutral=False
        assert res_df.iloc[2]['home_advantage'] == 0 # Neutral match
        
        # Check host advantage
        assert res_df.iloc[0]['host_advantage_home'] == 1 # Brazil is home in Brazil
        assert res_df.iloc[0]['host_advantage_away'] == 0
        assert res_df.iloc[2]['host_advantage_home'] == 0 # Croatia is home in Brazil
        assert res_df.iloc[2]['host_advantage_away'] == 0 # Mexico is away in Brazil (country != Mexico)
        
        # Check initial form
        assert res_df.iloc[0]['home_form'] == 0.5
        
        # Check updated form: Brazil won the first match (outcome = 1.0) on 2014-06-12.
        # Second match is on 2014-06-19 (7 days later).
        # Brazil form before second match: 1.0 (decayed by 7 days).
        # Since it's only one match in history, the weighted average is exactly 1.0 / 1.0 = 1.0.
        assert res_df.iloc[1]['home_form'] == 1.0
        
    finally:
        os.unlink(temp_path)
