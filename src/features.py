import math
import collections
import pandas as pd
import numpy as np
import os
from src.data_pipeline.curated_lookup import CuratedLookup

WC_YEARS = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
CONTINENTAL_TOURNAMENTS = ['UEFA Euro', 'Copa América', 'AFC Asian Cup', 'African Cup of Nations', 'Gold Cup', 'Oceania Nations Cup']

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

def get_squad_year(date):
    dt = pd.to_datetime(date)
    year = dt.year
    if year <= 2011:
        return 2010
    elif year <= 2015:
        return 2014
    elif year <= 2019:
        return 2018
    elif year <= 2023:
        return 2022
    else:
        return 2026

# Cache for squad features to make O(1) in main loop
_squad_cache = {}

def get_squad_features(team, year, squads_df):
    cache_key = (team, year)
    if cache_key in _squad_cache:
        return _squad_cache[cache_key]
        
    if squads_df is None or squads_df.empty:
        res = {"club_cohesion": 0.05, "peak_age_ratio": 0.50, "veteran_ratio": 0.15, "youth_ratio": 0.25}
        _squad_cache[cache_key] = res
        return res
        
    team_squad = squads_df[(squads_df['team'] == team) & (squads_df['year'] == year)]
    if team_squad.empty:
        res = {"club_cohesion": 0.05, "peak_age_ratio": 0.50, "veteran_ratio": 0.15, "youth_ratio": 0.25}
        _squad_cache[cache_key] = res
        return res
        
    N = len(team_squad)
    if N <= 1:
        cohesion = 0.0
    else:
        club_counts = team_squad['club'].value_counts()
        same_club_pairs = sum(c * (c - 1) for c in club_counts)
        cohesion = same_club_pairs / (N * (N - 1))
        
    peak = sum((team_squad['age'] >= 25) & (team_squad['age'] <= 29)) / N
    veteran = sum(team_squad['age'] > 31) / N
    youth = sum(team_squad['age'] < 23) / N
    
    res = {
        "club_cohesion": cohesion,
        "peak_age_ratio": peak,
        "veteran_ratio": veteran,
        "youth_ratio": youth
    }
    _squad_cache[cache_key] = res
    return res

def build_features(df: pd.DataFrame, curated_path: str = "data/curated_teams.json", half_life_days: float = 365.0, max_lookback: int = 10, squads_path: str = "data/squads.csv") -> pd.DataFrame:
    """
    Chronologically builds features for each match in the merged dataset.
    Maintains a running history of team outcomes to ensure O(N) calculation.
    """
    # Sort chronologically to prevent future leakage
    df = df.sort_values(by='date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize lookup systems
    lookup_system = CuratedLookup(curated_path)
    
    try:
        squads_df = pd.read_csv(squads_path)
    except Exception:
        squads_df = None
        
    # Track team histories
    team_history = collections.defaultdict(list) # list of (date, outcome_score)
    team_elo_history = collections.defaultdict(list) # list of (date, elo)
    team_wc_campaigns = collections.defaultdict(lambda: collections.defaultdict(int)) # team -> {year: match_count}
    team_continental_campaigns = collections.defaultdict(list) # team -> list of dicts: {'date': date, 'tournament': tournament, 'year': year}
    
    # Result containers
    feats = {
        'elo_diff': [], 'home_elo': [], 'away_elo': [],
        'home_form': [], 'away_form': [], 'form_diff': [],
        'home_squad_quality': [], 'away_squad_quality': [], 'squad_quality_diff': [],
        'home_interim_coach': [], 'away_interim_coach': [],
        'home_advantage': [], 'host_advantage_home': [], 'host_advantage_away': [],
        
        # New features
        'home_cohesion': [], 'away_cohesion': [], 'cohesion_diff': [],
        'home_peak_age_ratio': [], 'away_peak_age_ratio': [], 'peak_age_ratio_diff': [],
        'home_veteran_ratio': [], 'away_veteran_ratio': [], 'veteran_ratio_diff': [],
        'home_youth_ratio': [], 'away_youth_ratio': [], 'youth_ratio_diff': [],
        'home_tournament_experience': [], 'away_tournament_experience': [], 'tournament_experience_diff': [],
        'home_momentum': [], 'away_momentum': [], 'momentum_diff': []
    }
    targets = []
    
    for idx, row in df.iterrows():
        date = row['date']
        home = row['home_team']
        away = row['away_team']
        home_elo = float(row['home_elo_before'])
        away_elo = float(row['away_elo_before'])
        neutral = bool(row['neutral'])
        country = row['country']
        tournament = row.get('tournament', 'Friendly')
        match_year = date.year
        
        # 1. Elo features
        feats['home_elo'].append(home_elo)
        feats['away_elo'].append(away_elo)
        feats['elo_diff'].append(home_elo - away_elo)
        
        # 2. Form features (Exponential Decay)
        home_f = compute_decay_form(team_history[home], date, half_life_days, max_lookback)
        away_f = compute_decay_form(team_history[away], date, half_life_days, max_lookback)
        feats['home_form'].append(home_f)
        feats['away_form'].append(away_f)
        feats['form_diff'].append(home_f - away_f)
        
        # 3. Curated features: coach/squad
        home_curated = lookup_system.lookup(home, date)
        away_curated = lookup_system.lookup(away, date)
        
        feats['home_squad_quality'].append(home_curated['squad_quality'])
        feats['away_squad_quality'].append(away_curated['squad_quality'])
        feats['squad_quality_diff'].append(home_curated['squad_quality'] - away_curated['squad_quality'])
        
        feats['home_interim_coach'].append(1 if home_curated['is_interim'] else 0)
        feats['away_interim_coach'].append(1 if away_curated['is_interim'] else 0)
        
        # 4. Host/Home Advantage
        feats['home_advantage'].append(1 if (not neutral and country == home) else 0)
        feats['host_advantage_home'].append(1 if country == home else 0)
        feats['host_advantage_away'].append(1 if country == away else 0)
        
        # 5. Squad Cohesion and Age features
        sq_year = get_squad_year(date)
        home_sq = get_squad_features(home, sq_year, squads_df)
        away_sq = get_squad_features(away, sq_year, squads_df)
        
        feats['home_cohesion'].append(home_sq['club_cohesion'])
        feats['away_cohesion'].append(away_sq['club_cohesion'])
        feats['cohesion_diff'].append(home_sq['club_cohesion'] - away_sq['club_cohesion'])
        
        feats['home_peak_age_ratio'].append(home_sq['peak_age_ratio'])
        feats['away_peak_age_ratio'].append(away_sq['peak_age_ratio'])
        feats['peak_age_ratio_diff'].append(home_sq['peak_age_ratio'] - away_sq['peak_age_ratio'])
        
        feats['home_veteran_ratio'].append(home_sq['veteran_ratio'])
        feats['away_veteran_ratio'].append(away_sq['veteran_ratio'])
        feats['veteran_ratio_diff'].append(home_sq['veteran_ratio'] - away_sq['veteran_ratio'])
        
        feats['home_youth_ratio'].append(home_sq['youth_ratio'])
        feats['away_youth_ratio'].append(away_sq['youth_ratio'])
        feats['youth_ratio_diff'].append(home_sq['youth_ratio'] - away_sq['youth_ratio'])
        
        # 6. Tournament Experience Index
        def get_experience(team):
            completed_years = [y for y in team_wc_campaigns[team].keys() if y < match_year]
            if not completed_years:
                return 0.0
            recent_years = completed_years[-3:] if len(completed_years) >= 3 else completed_years
            total_wc = 0
            recent_wc = 0
            total_ko = 0
            total_qf = 0
            total_sf = 0
            for y in completed_years:
                cnt = team_wc_campaigns[team][y]
                if cnt >= 1:
                    total_wc += 1
                    if y in recent_years:
                        recent_wc += 1
                    if cnt >= 4:
                        total_ko += 1
                    if cnt >= 5:
                        total_qf += 1
                    if cnt >= 6:
                        total_sf += 1
            # Weighted index
            return (1.0 * total_wc) + (2.0 * recent_wc) + (2.0 * total_ko) + (3.0 * total_qf) + (4.0 * total_sf)
            
        home_exp = get_experience(home)
        away_exp = get_experience(away)
        feats['home_tournament_experience'].append(home_exp)
        feats['away_tournament_experience'].append(away_exp)
        feats['tournament_experience_diff'].append(home_exp - away_exp)
        
        # 7. Momentum Score
        def get_momentum(team, current_elo):
            # A. Elo trend over last 365 days
            elo_hist = team_elo_history[team]
            elo_trend = 0.0
            if elo_hist:
                target_date = date - pd.Timedelta(days=365)
                elo_365 = elo_hist[0][1] # default to oldest if none before target
                for d, elo in reversed(elo_hist):
                    if d <= target_date:
                        elo_365 = elo
                        break
                elo_trend = current_elo - elo_365
            elo_trend_norm = np.clip((elo_trend + 200.0) / 400.0, 0.0, 1.0)
            
            # B. Recent form (last 20 matches win rate)
            hist = team_history[team]
            if not hist:
                form_20 = 0.5
            else:
                outcomes = [out for _, out in hist[-20:]]
                form_20 = sum(outcomes) / len(outcomes)
                
            # C. Continental Tournament performance in last 4 years
            cont_hist = team_continental_campaigns[team]
            cont_perf = 0.0
            if cont_hist:
                four_years_ago = date - pd.Timedelta(days=1460)
                recent_cont = [m for m in cont_hist if four_years_ago <= m['date'] < date]
                if recent_cont:
                    campaigns = collections.defaultdict(int)
                    for m in recent_cont:
                        campaigns[(m['tournament'], m['year'])] += 1
                    max_matches = max(campaigns.values())
                    if max_matches >= 6:
                        cont_perf = 1.0
                    elif max_matches == 5:
                        cont_perf = 0.8
                    elif max_matches == 4:
                        cont_perf = 0.5
                    else:
                        cont_perf = 0.2
                        
            # D. Unbeaten streak
            unbeaten = 0
            if hist:
                for _, out in reversed(hist):
                    if out >= 0.5:
                        unbeaten += 1
                    else:
                        break
            unbeaten_norm = min(unbeaten / 15.0, 1.0)
            
            # Combined momentum_score: 40% Elo Trend + 30% Recent Form + 20% Continental + 10% Unbeaten Streak
            return 0.4 * elo_trend_norm + 0.3 * form_20 + 0.2 * cont_perf + 0.1 * unbeaten_norm
            
        home_mom = get_momentum(home, home_elo)
        away_mom = get_momentum(away, away_elo)
        feats['home_momentum'].append(home_mom)
        feats['away_momentum'].append(away_mom)
        feats['momentum_diff'].append(home_mom - away_mom)
        
        # 8. Target variable: 0 = Home Win, 1 = Draw, 2 = Away Win
        hs = row['home_score']
        as_ = row['away_score']
        if hs > as_:
            target = 0
        elif hs == as_:
            target = 1
        else:
            target = 2
        targets.append(target)
        
        # 9. Update chronological states AFTER calculating features for current match
        home_outcome = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        away_outcome = 1.0 if as_ > hs else (0.5 if hs == as_ else 0.0)
        
        team_history[home].append((date, home_outcome))
        team_history[away].append((date, away_outcome))
        
        team_elo_history[home].append((date, home_elo))
        team_elo_history[away].append((date, away_elo))
        
        if tournament == 'FIFA World Cup':
            team_wc_campaigns[home][match_year] += 1
            team_wc_campaigns[away][match_year] += 1
            
        if tournament in CONTINENTAL_TOURNAMENTS:
            team_continental_campaigns[home].append({'date': date, 'tournament': tournament, 'year': match_year})
            team_continental_campaigns[away].append({'date': date, 'tournament': tournament, 'year': match_year})
            
    # Add new features to a copy of the dataframe
    res_df = df.copy()
    for col_name, value_list in feats.items():
        res_df[col_name] = value_list
    res_df['target'] = targets
    
    return res_df
