import os
import json
import datetime
import bisect
import pandas as pd
import numpy as np

def parse_elo_tsv(filepath: str) -> pd.DataFrame:
    """
    Parses a team's Elo TSV file from eloratings.net robustly.
    Handles delimiter detection, header detection, and date normalization.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Elo TSV file not found: {filepath}")
        
    # Read the first few lines to detect delimiter and headers
    with open(filepath, 'r', encoding='utf-8') as f:
        sample_lines = [f.readline() for _ in range(5)]
        
    if not sample_lines:
        # Empty file
        return pd.DataFrame()

    # Delimiter detection: tab vs comma
    tab_count = sum(line.count('\t') for line in sample_lines)
    comma_count = sum(line.count(',') for line in sample_lines)
    sep = '\t' if tab_count >= comma_count else ','
    
    # Try reading without headers first
    try:
        df = pd.read_csv(filepath, sep=sep, header=None)
    except Exception as e:
        raise ValueError(f"Error reading Elo TSV file {filepath}: {e}")
        
    # Header detection
    first_row = df.iloc[0]
    try:
        int(first_row.iloc[0])
        int(first_row.iloc[1])
        int(first_row.iloc[2])
        has_header = False
    except (ValueError, TypeError, IndexError):
        has_header = True
        
    if has_header:
        df = pd.read_csv(filepath, sep=sep, header=0)
        
    expected_col_count = 16
    if df.shape[1] < 12:
        raise ValueError(f"Unexpected number of columns in Elo TSV {filepath}: {df.shape[1]}")
        
    df = df.iloc[:, :expected_col_count]
    
    col_names = [
        'year', 'month', 'day', 'team_a_code', 'team_b_code',
        'team_a_goals', 'team_b_goals', 'tournament', 'neutral_country',
        'points_change', 'team_a_elo_after', 'team_b_elo_after',
        'team_a_rank_change', 'team_b_rank_change', 'team_a_rank_after', 'team_b_rank_after'
    ]
    df.columns = col_names[:df.shape[1]]
    
    # Clean up points_change column
    if 'points_change' in df.columns:
        df['points_change'] = df['points_change'].astype(str).str.replace('−', '-').str.replace('+', '', regex=False)
        df['points_change'] = pd.to_numeric(df['points_change'], errors='coerce').fillna(0).astype(int)
        
    # Normalize date
    df['year'] = df['year'].astype(str)
    df['month'] = df['month'].astype(str).str.zfill(2)
    df['day'] = df['day'].astype(str).str.zfill(2)
    df['date'] = df['year'] + '-' + df['month'] + '-' + df['day']
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    
    df = df.dropna(subset=['date'])
    
    return df

def merge_elo_and_results(results_path: str, canonical_teams_path: str, elo_dir: str) -> pd.DataFrame:
    """
    Merges Jürisoo's results dataset with the historical Elo ratings.
    Reconstructs the before-match Elo values for both teams.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not os.path.exists(canonical_teams_path):
        raise FileNotFoundError(f"Canonical teams file not found: {canonical_teams_path}")
    if not os.path.exists(elo_dir):
        raise FileNotFoundError(f"Elo directory not found: {elo_dir}")
        
    # 1. Load canonical teams mapping
    with open(canonical_teams_path, 'r', encoding='utf-8') as f:
        canonical_mapping = json.load(f)
        
    # 2. Load results
    results_df = pd.read_csv(results_path)
    results_df['date'] = pd.to_datetime(results_df['date'])
    # Filter for matches >= 2000-01-01
    results_df = results_df[results_df['date'] >= '2000-01-01'].copy()
    
    # 3. Load all Elo histories
    global_matches = {}
    team_elo_history = {}
    
    print("Loading Elo histories...")
    for team_name, details in canonical_mapping.items():
        code = details['code']
        elo_file = details['elo_file']
        filepath = os.path.join(elo_dir, elo_file)
        if not os.path.exists(filepath):
            continue
            
        try:
            df_elo = parse_elo_tsv(filepath)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            continue
            
        history_list = []
        for idx, row in df_elo.iterrows():
            d = row['date'].date()
            ta = row['team_a_code']
            tb = row['team_b_code']
            pa = row['team_a_elo_after']
            pb = row['team_b_elo_after']
            pts = row['points_change']
            
            # Add to global_matches
            global_matches[(d, ta, tb)] = (pa, pb, pts)
            
            # Determine if this team is Team A or Team B
            if ta == code:
                history_list.append((d, pa))
            elif tb == code:
                history_list.append((d, pb))
                
        # Sort history by date and store
        history_list.sort(key=lambda x: x[0])
        team_elo_history[code] = history_list
        
    # 4. Helper for looking up Elo before a match
    def get_elo_ratings(date_val, h_code, a_code):
        d = date_val.date()
        # Check direct matches in global_matches for D, D-1, D+1
        for offset in [0, -1, 1]:
            target_d = d + datetime.timedelta(days=offset)
            
            # Case 1: Home is Team A, Away is Team B
            if (target_d, h_code, a_code) in global_matches:
                pa, pb, pts = global_matches[(target_d, h_code, a_code)]
                return pa - pts, pb + pts
                
            # Case 2: Away is Team A, Home is Team B
            if (target_d, a_code, h_code) in global_matches:
                pa, pb, pts = global_matches[(target_d, a_code, h_code)]
                return pb + pts, pa - pts
                
        # Fallback to temporal lookup
        h_elo = get_temporal_elo(h_code, d)
        a_elo = get_temporal_elo(a_code, d)
        return h_elo, a_elo

    def get_temporal_elo(code, d):
        if code not in team_elo_history or not team_elo_history[code]:
            return 1500.0
        hist = team_elo_history[code]
        # Binary search
        dates = [x[0] for x in hist]
        idx = bisect.bisect_left(dates, d)
        if idx > 0:
            return hist[idx - 1][1]
        else:
            return hist[0][1]

    # 5. Map results to canonical codes
    home_codes = []
    away_codes = []
    home_elos = []
    away_elos = []
    valid_mask = []
    
    for idx, row in results_df.iterrows():
        h_name = row['home_team']
        a_name = row['away_team']
        
        if h_name in canonical_mapping and a_name in canonical_mapping:
            h_code = canonical_mapping[h_name]['code']
            a_code = canonical_mapping[a_name]['code']
            
            h_elo, a_elo = get_elo_ratings(row['date'], h_code, a_code)
            
            home_codes.append(h_code)
            away_codes.append(a_code)
            home_elos.append(h_elo)
            away_elos.append(a_elo)
            valid_mask.append(True)
        else:
            home_codes.append(None)
            away_codes.append(None)
            home_elos.append(None)
            away_elos.append(None)
            valid_mask.append(False)
            
    results_df['home_code'] = home_codes
    results_df['away_code'] = away_codes
    results_df['home_elo_before'] = home_elos
    results_df['away_elo_before'] = away_elos
    
    # Keep only matches where both teams were successfully mapped
    merged_df = results_df[valid_mask].copy()
    
    # Cast Elo columns to float/numeric
    merged_df['home_elo_before'] = pd.to_numeric(merged_df['home_elo_before'])
    merged_df['away_elo_before'] = pd.to_numeric(merged_df['away_elo_before'])
    
    print(f"Merged {len(merged_df)} matches successfully.")
    return merged_df
