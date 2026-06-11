import math
import collections
import pandas as pd
import numpy as np
from src.data_pipeline.curated_lookup import CuratedLookup

def compute_decay_form(history, current_date, half_life_days, max_lookback=10):
    """
    Computes the exponential decay form of a team based on previous match outcomes.
    Win = 1.0, Draw = 0.5, Loss = 0.0.
    """
    if not history:
        return 0.5
        
    recent = history[-max_lookback:]
    total_weight = 0.0
    weighted_outcome = 0.0
    
    # lambda = ln(2) / half_life
    lam = 0.69314718056 / half_life_days
    
    for match_date, outcome in recent:
        days = (current_date - match_date).days
        if days < 0:
            days = 0
        weight = math.exp(-lam * days)
        weighted_outcome += outcome * weight
        total_weight += weight
        
    if total_weight == 0.0:
        return 0.5
    return weighted_outcome / total_weight

def build_features(df: pd.DataFrame, curated_path: str = "data/curated_teams.json", half_life_days: float = 365.0, max_lookback: int = 10) -> pd.DataFrame:
    """
    Chronologically builds features for each match in the merged dataset.
    Maintains a running history of team outcomes to ensure O(N) calculation of form.
    """
    # Sort chronologically to prevent future leakage
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Initialize lookup system
    lookup_system = CuratedLookup(curated_path)
    
    # Track team histories: list of (date, outcome_score)
    team_history = collections.defaultdict(list)
    
    # Result container lists
    elo_diffs = []
    home_elos = []
    away_elos = []
    home_forms = []
    away_forms = []
    form_diffs = []
    home_squad_quals = []
    away_squad_quals = []
    squad_qual_diffs = []
    home_interims = []
    away_interims = []
    home_advantages = []
    host_adv_homes = []
    host_adv_aways = []
    targets = []
    
    for idx, row in df.iterrows():
        date = row['date']
        home = row['home_team']
        away = row['away_team']
        home_elo = float(row['home_elo_before'])
        away_elo = float(row['away_elo_before'])
        neutral = bool(row['neutral'])
        country = row['country']
        
        # 1. Elo features
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        elo_diffs.append(home_elo - away_elo)
        
        # 2. Form features
        home_f = compute_decay_form(team_history[home], date, half_life_days, max_lookback)
        away_f = compute_decay_form(team_history[away], date, half_life_days, max_lookback)
        home_forms.append(home_f)
        away_forms.append(away_f)
        form_diffs.append(home_f - away_f)
        
        # 3. Curated features: coach/squad
        home_curated = lookup_system.lookup(home, date)
        away_curated = lookup_system.lookup(away, date)
        
        home_squad_quals.append(home_curated['squad_quality'])
        away_squad_quals.append(away_curated['squad_quality'])
        squad_qual_diffs.append(home_curated['squad_quality'] - away_curated['squad_quality'])
        
        home_interims.append(1 if home_curated['is_interim'] else 0)
        away_interims.append(1 if away_curated['is_interim'] else 0)
        
        # 4. Host/Home Advantage
        # Home advantage: playing at home and not neutral
        home_advantages.append(1 if (not neutral and country == home) else 0)
        
        # Host advantage: tournament played in team's country (can apply even if tournament match is marked neutral)
        host_adv_homes.append(1 if country == home else 0)
        host_adv_aways.append(1 if country == away else 0)
        
        # 5. Target variable: 0 = Home Win, 1 = Draw, 2 = Away Win
        hs = row['home_score']
        as_ = row['away_score']
        if hs > as_:
            target = 0
        elif hs == as_:
            target = 1
        else:
            target = 2
        targets.append(target)
        
        # 6. Update histories with this match outcome
        home_outcome = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        away_outcome = 1.0 if as_ > hs else (0.5 if hs == as_ else 0.0)
        
        team_history[home].append((date, home_outcome))
        team_history[away].append((date, away_outcome))
        
    # Add new features to a copy of the dataframe
    res_df = df.copy()
    res_df['elo_diff'] = elo_diffs
    res_df['home_elo'] = home_elos
    res_df['away_elo'] = away_elos
    res_df['home_form'] = home_forms
    res_df['away_form'] = away_forms
    res_df['form_diff'] = form_diffs
    res_df['home_squad_quality'] = home_squad_quals
    res_df['away_squad_quality'] = away_squad_quals
    res_df['squad_quality_diff'] = squad_qual_diffs
    res_df['home_interim_coach'] = home_interims
    res_df['away_interim_coach'] = away_interims
    res_df['home_advantage'] = home_advantages
    res_df['host_advantage_home'] = host_adv_homes
    res_df['host_advantage_away'] = host_adv_aways
    res_df['target'] = targets
    
    return res_df
