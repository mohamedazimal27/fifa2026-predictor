import os
import pickle
import json
import random
import collections
import math
import functools
import numpy as np
import pandas as pd
from src.data_pipeline.curated_lookup import CuratedLookup
from src.data_pipeline.data_loader import parse_elo_tsv
from src.third_place_router import assign_third_places
from src.features import get_squad_features, CONTINENTAL_TOURNAMENTS

# 2026 World Cup Groups and Teams
# Sources: Official FIFA draw + Kaggle dataset (areezvisram12/fifa-world-cup-2026-match-data-unofficial)
# Teams listed in order T0, T1, T2, T3 — must match GROUP_FIXTURE_PATTERNS indices below
# Playoff placeholders are mapped to best-guess teams for simulation purposes:
#   UEFA Playoff D (Gp A) -> Czechia, UEFA Playoff A (Gp B) -> Bosnia_and_Herzegovina
#   UEFA Playoff C (Gp D) -> Turkey,  UEFA Playoff B (Gp F) -> Sweden
#   FIFA Playoff 2 (Gp I) -> Iraq,    FIFA Playoff 1 (Gp K) -> DR_Congo
GROUPS_2026 = {
    'A': ['Mexico', 'South_Africa', 'South_Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia_and_Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United_States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory_Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New_Zealand'],
    'H': ['Spain', 'Cape_Verde', 'Saudi_Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR_Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}

# Official FIFA 2026 fixture order per group, expressed as (T_home_idx, T_away_idx) pairs.
# Derived from Kaggle dataset: areezvisram12/fifa-world-cup-2026-match-data-unofficial
# MD = Matchday. Each group plays 6 matches (round-robin of 4 teams).
# Pattern A/B/C/I: MD1=(0,1),(2,3) | MD2=(3,1),(0,2) | MD3=(3,0),(1,2)
# Pattern D/E/F/G/H/J/K/L: MD1=(0,1),(2,3) | MD2=(0,2),(3,1) | MD3=(some vary, see below)
GROUP_FIXTURE_PATTERNS = {
    # Groups A, B, C follow the same pattern (Matchday 3: playoff vs T0, then T1 vs T2)
    'A': [(0,1),(2,3),(3,1),(0,2),(3,0),(1,2)],
    'B': [(0,1),(2,3),(3,1),(0,2),(3,0),(1,2)],
    'C': [(0,1),(2,3),(3,1),(0,2),(3,0),(1,2)],
    # Groups D, E, F, G, H, J differ: MD2 starts with (0,2) instead of (3,1)
    'D': [(0,1),(2,3),(0,2),(3,1),(3,0),(1,2)],
    'E': [(0,1),(2,3),(0,2),(3,1),(1,2),(3,0)],
    'F': [(0,1),(2,3),(0,2),(3,1),(1,2),(3,0)],
    'G': [(0,1),(2,3),(0,2),(3,1),(1,2),(3,0)],
    'H': [(0,1),(2,3),(0,2),(3,1),(1,2),(3,0)],
    'I': [(0,1),(2,3),(0,2),(3,1),(3,0),(1,2)],
    'J': [(0,1),(2,3),(0,2),(3,1),(1,2),(3,0)],
    'K': [(0,1),(2,3),(0,2),(3,1),(3,0),(1,2)],
    'L': [(0,1),(2,3),(0,2),(3,1),(3,0),(1,2)],
}

# Real-world results for matches that have already been played at FIFA WC 2026.
# Updated as the tournament progresses. Format: (home, away) -> (home_goals, away_goals)
REAL_WORLD_RESULTS = {
    # Matchday 1 — June 11, 2026
    ("Mexico", "South_Africa"): (2, 0),
    ("South_Korea", "Czechia"): (2, 1),
}


def compare_teams(a, b, group_matches):
    """
    FIFA World Cup tiebreaker sorting:
    1. Points
    2. Goal Difference (GD)
    3. Goals Scored (GS)
    4. Head-to-Head Points
    5. Head-to-Head GD
    6. Head-to-Head GS
    7. Seeded Draw / Elo fallback
    """
    t1 = a['team']
    t2 = b['team']
    
    # 1. Points
    if a['points'] != b['points']:
        return a['points'] - b['points']
        
    # 2. GD
    if a['gd'] != b['gd']:
        return a['gd'] - b['gd']
        
    # 3. GS
    if a['gs'] != b['gs']:
        return a['gs'] - b['gs']
        
    # 4. H2H match check
    h2h_match = None
    for h, aw, h_g, a_g in group_matches:
        if (h == t1 and aw == t2):
            h2h_match = (h_g, a_g)
            break
        elif (h == t2 and aw == t1):
            h2h_match = (a_g, h_g)
            break
            
    if h2h_match is not None:
        g1, g2 = h2h_match
        # H2H Points
        p1 = 3 if g1 > g2 else (1 if g1 == g2 else 0)
        p2 = 3 if g2 > g1 else (1 if g1 == g2 else 0)
        if p1 != p2:
            return p1 - p2
        # H2H GD
        gd1 = g1 - g2
        gd2 = g2 - g1
        if gd1 != gd2:
            return gd1 - gd2
        # H2H GS
        if g1 != g2:
            return g1 - g2
            
    # 5. Elo rating fallback
    if a['elo'] != b['elo']:
        return a['elo'] - b['elo']
        
    return 0

class TournamentSimulator:
    def __init__(self, model_path="models/fifa_model.pkl", curated_path="data/curated_teams.json", elo_dir="data/elo", canonical_path="data/canonical_teams.json", results_path="data/results.csv", squads_path="data/squads.csv"):
        self.curated_path = curated_path
        self.elo_dir = elo_dir
        self.squads_path = squads_path
        
        # Load pre-trained model
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        self.base_model = model_dict["base_model"]
        self.calibrator = model_dict["calibrator"]
        self.feature_cols = model_dict["feature_cols"]
        
        # Load curated lookup system
        self.lookup_system = CuratedLookup(curated_path)
        
        # Load canonical mapping
        with open(canonical_path, 'r', encoding='utf-8') as f:
            self.canonical_mapping = json.load(f)
            
        # Load raw datasets for dynamic feature precomputation
        results_df = pd.read_csv(results_path)
        results_df['date'] = pd.to_datetime(results_df['date'])
        
        try:
            squads_df = pd.read_csv(squads_path)
        except Exception:
            squads_df = None
            
        # Get starting Elo ratings for all 2026 teams
        self.starting_elos = {}
        for group, teams in GROUPS_2026.items():
            for team in teams:
                self.starting_elos[team] = self._get_latest_elo(team)
                
        # Precompute team features using exact historical formulas up to June 11, 2026
        self.team_features = {}
        for group, teams in GROUPS_2026.items():
            for team in teams:
                self.team_features[team] = self._precompute_team_features(team, results_df, squads_df)
                
        # Pre-compute match probabilities cache for all 48 teams
        self.matchup_cache = {}
        self._precompute_matchups()

    def _canonical_name(self, team_name):
        name_spaced = team_name.replace('_', ' ')
        if name_spaced in self.canonical_mapping:
            return name_spaced
        custom_mappings = {
            "Czechia": "Czech Republic",
            "Curacao": "Curaçao",
            "Congo DR": "DR Congo",
            "DR Congo": "DR Congo",
            "Congo": "DR Congo"
        }
        if name_spaced in custom_mappings:
            return custom_mappings[name_spaced]
        if team_name in custom_mappings:
            return custom_mappings[team_name]
        return name_spaced

    def _get_latest_elo(self, team_name):
        canonical_name = self._canonical_name(team_name)
        if canonical_name not in self.canonical_mapping:
            return 1500.0
        code = self.canonical_mapping[canonical_name]['code']
        elo_file = self.canonical_mapping[canonical_name]['elo_file']
        tsv_path = os.path.join(self.elo_dir, elo_file)
        if not os.path.exists(tsv_path):
            return 1500.0
        df_elo = parse_elo_tsv(tsv_path)
        if df_elo.empty:
            return 1500.0
        last_row = df_elo.iloc[-1]
        ta = last_row['team_a_code']
        tb = last_row['team_b_code']
        pa = last_row['team_a_elo_after']
        pb = last_row['team_b_elo_after']
        if ta == code:
            return float(pa)
        elif tb == code:
            return float(pb)
        return 1500.0

    def _precompute_team_features(self, team, results_df, squads_df, date_str="2026-06-11"):
        date_cutoff = pd.to_datetime(date_str)
        canonical_name = self._canonical_name(team)
        
        # 1. World Cup Experience Index
        wc_matches = results_df[
            ((results_df['home_team'] == canonical_name) | (results_df['away_team'] == canonical_name)) &
            (results_df['tournament'] == 'FIFA World Cup') &
            (results_df['date'] < date_cutoff)
        ]
        wc_years = wc_matches['date'].dt.year.unique()
        wc_counts = {}
        for y in wc_years:
            wc_counts[y] = len(wc_matches[wc_matches['date'].dt.year == y])
            
        total_wc = len(wc_counts)
        recent_wc = sum(1 for y in wc_counts if y in [2014, 2018, 2022])
        total_ko = sum(1 for y, c in wc_counts.items() if c >= 4)
        total_qf = sum(1 for y, c in wc_counts.items() if c >= 5)
        total_sf = sum(1 for y, c in wc_counts.items() if c >= 6)
        experience = (1.0 * total_wc) + (2.0 * recent_wc) + (2.0 * total_ko) + (3.0 * total_qf) + (4.0 * total_sf)
        
        # 2. Squad Features
        sq_feats = get_squad_features(canonical_name, 2026, squads_df)
        
        # 3. Momentum Score
        # A. Elo Trend over last 365 days
        current_elo = self.starting_elos.get(team, 1500.0)
        elo_365 = current_elo
        if canonical_name in self.canonical_mapping:
            code = self.canonical_mapping[canonical_name]['code']
            elo_file = self.canonical_mapping[canonical_name]['elo_file']
            tsv_path = os.path.join(self.elo_dir, elo_file)
            if os.path.exists(tsv_path):
                df_elo = parse_elo_tsv(tsv_path)
                if not df_elo.empty:
                    df_elo['date'] = pd.to_datetime(df_elo['date'])
                    target_date = date_cutoff - pd.Timedelta(days=365)
                    past_rows = df_elo[df_elo['date'] <= target_date]
                    if not past_rows.empty:
                        last_row = past_rows.iloc[-1]
                        ta = last_row['team_a_code']
                        tb = last_row['team_b_code']
                        pa = last_row['team_a_elo_after']
                        pb = last_row['team_b_elo_after']
                        if ta == code:
                            elo_365 = float(pa)
                        elif tb == code:
                            elo_365 = float(pb)
        elo_trend = current_elo - elo_365
        elo_trend_norm = np.clip((elo_trend + 200.0) / 400.0, 0.0, 1.0)
        
        # B. Recent Form (last 20 matches win rate before cutoff)
        team_matches = results_df[
            ((results_df['home_team'] == canonical_name) | (results_df['away_team'] == canonical_name)) &
            (results_df['date'] < date_cutoff)
        ].sort_values(by='date')
        
        outcomes = []
        for _, row in team_matches.tail(20).iterrows():
            hs = row['home_score']
            as_ = row['away_score']
            is_home = (row['home_team'] == canonical_name)
            if hs == as_:
                outcomes.append(0.5)
            elif (hs > as_ and is_home) or (as_ > hs and not is_home):
                outcomes.append(1.0)
            else:
                outcomes.append(0.0)
        form_20 = sum(outcomes) / len(outcomes) if outcomes else 0.5
        
        # C. Continental Tournament performance in last 4 years
        four_years_ago = date_cutoff - pd.Timedelta(days=1460)
        cont_matches = results_df[
            ((results_df['home_team'] == canonical_name) | (results_df['away_team'] == canonical_name)) &
            (results_df['tournament'].isin(CONTINENTAL_TOURNAMENTS)) &
            (results_df['date'] >= four_years_ago) &
            (results_df['date'] < date_cutoff)
        ]
        cont_perf = 0.0
        if not cont_matches.empty:
            campaigns = collections.defaultdict(int)
            for _, row in cont_matches.iterrows():
                campaigns[(row['tournament'], row['date'].year)] += 1
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
        for _, row in team_matches.iloc[::-1].iterrows():
            hs = row['home_score']
            as_ = row['away_score']
            is_home = (row['home_team'] == canonical_name)
            if hs == as_:
                unbeaten += 1
            elif (hs > as_ and is_home) or (as_ > hs and not is_home):
                unbeaten += 1
            else:
                break
        unbeaten_norm = min(unbeaten / 15.0, 1.0)
        
        # E. Final Momentum calculation
        momentum = 0.4 * elo_trend_norm + 0.3 * form_20 + 0.2 * cont_perf + 0.1 * unbeaten_norm
        
        # Curated features (coach/squad quality)
        curated = self.lookup_system.lookup(canonical_name, date_str)
        
        return {
            'elo': current_elo,
            'form': form_20,
            'squad_quality': curated['squad_quality'],
            'is_interim': 1 if curated['is_interim'] else 0,
            'host_advantage': 1 if team in ['United_States', 'Mexico', 'Canada'] else 0,
            
            # New features
            'cohesion': sq_feats['club_cohesion'],
            'peak_age_ratio': sq_feats['peak_age_ratio'],
            'veteran_ratio': sq_feats['veteran_ratio'],
            'youth_ratio': sq_feats['youth_ratio'],
            'tournament_experience': experience,
            'momentum': momentum
        }

    def _precompute_matchups(self):
        """Pre-computes and caches probabilities for all 48 teams in pairwise matchups."""
        teams = list(self.starting_elos.keys())
        n_teams = len(teams)
        
        rows = []
        keys = []
        
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                t1, t2 = teams[i], teams[j]
                
                f1 = self.team_features[t1]
                f2 = self.team_features[t2]
                
                for fatigue1 in range(6):
                    for fatigue2 in range(6):
                        elo1 = f1['elo'] - fatigue1 * 20.0
                        elo2 = f2['elo'] - fatigue2 * 20.0
                        
                        elo_diff = elo1 - elo2
                        form_diff = f1['form'] - f2['form']
                        squad_diff = f1['squad_quality'] - f2['squad_quality']
                        
                        host_advantage_home = 1 if f1['host_advantage'] == 1 and f2['host_advantage'] == 0 else 0
                        host_advantage_away = 1 if f2['host_advantage'] == 1 and f1['host_advantage'] == 0 else 0
                        
                        row = {
                            'elo_diff': elo_diff,
                            'home_elo': elo1,
                            'away_elo': elo2,
                            'home_form': f1['form'],
                            'away_form': f2['form'],
                            'form_diff': form_diff,
                            'home_squad_quality': f1['squad_quality'],
                            'away_squad_quality': f2['squad_quality'],
                            'squad_quality_diff': squad_diff,
                            'home_interim_coach': f1['is_interim'],
                            'away_interim_coach': f2['is_interim'],
                            'home_advantage': f1['host_advantage'],
                            'host_advantage_home': host_advantage_home,
                            'host_advantage_away': host_advantage_away,
                            
                            # New features
                            'home_cohesion': f1['cohesion'],
                            'away_cohesion': f2['cohesion'],
                            'cohesion_diff': f1['cohesion'] - f2['cohesion'],
                            
                            'home_peak_age_ratio': f1['peak_age_ratio'],
                            'away_peak_age_ratio': f2['peak_age_ratio'],
                            'peak_age_ratio_diff': f1['peak_age_ratio'] - f2['peak_age_ratio'],
                            
                            'home_veteran_ratio': f1['veteran_ratio'],
                            'away_veteran_ratio': f2['veteran_ratio'],
                            'veteran_ratio_diff': f1['veteran_ratio'] - f2['veteran_ratio'],
                            
                            'home_youth_ratio': f1['youth_ratio'],
                            'away_youth_ratio': f2['youth_ratio'],
                            'youth_ratio_diff': f1['youth_ratio'] - f2['youth_ratio'],
                            
                            'home_tournament_experience': f1['tournament_experience'],
                            'away_tournament_experience': f2['tournament_experience'],
                            'tournament_experience_diff': f1['tournament_experience'] - f2['tournament_experience'],
                            
                            'home_momentum': f1['momentum'],
                            'away_momentum': f2['momentum'],
                            'momentum_diff': f1['momentum'] - f2['momentum']
                        }
                        rows.append(row)
                        keys.append((t1, t2, fatigue1, fatigue2))
                
        df_batch = pd.DataFrame(rows)[self.feature_cols]
        base_preds = self.base_model.predict_proba(df_batch)
        cal_preds = self.calibrator.predict_proba(base_preds)
        
        for key, pred in zip(keys, cal_preds):
            pred = np.array(pred, dtype=np.float64)
            pred = pred / pred.sum()
            self.matchup_cache[key] = pred

    def apply_injury_shocks(self, shocks):
        """
        Applies injury shocks to starting Elos and squad qualities, then rebuilds matchups.
        shocks: dict of {team: importance_tier} where importance_tier is 'key', 'world_class', or 'indispensable'
        """
        # Re-initialize to fetch starting values
        results_df = pd.read_csv("data/results.csv")
        results_df['date'] = pd.to_datetime(results_df['date'])
        try:
            squads_df = pd.read_csv(self.squads_path)
        except Exception:
            squads_df = None
            
        for team in self.starting_elos.keys():
            self.starting_elos[team] = self._get_latest_elo(team)
            self.team_features[team] = self._precompute_team_features(team, results_df, squads_df)
            
        if not shocks:
            self._precompute_matchups()
            return
            
        for team, tier in shocks.items():
            if team not in self.starting_elos:
                continue
            if tier == 'key':
                elo_drop = 30.0
                sq_drop = 0.05
            elif tier == 'world_class':
                elo_drop = 50.0
                sq_drop = 0.10
            elif tier == 'indispensable':
                elo_drop = 80.0
                sq_drop = 0.15
            else:
                continue
                
            self.starting_elos[team] -= elo_drop
            self.team_features[team]['elo'] -= elo_drop
            self.team_features[team]['squad_quality'] = max(0.0, self.team_features[team]['squad_quality'] - sq_drop)
            
        self._precompute_matchups()

    def predict_match(self, team1, team2, fatigue1=0, fatigue2=0):
        """
        Predicts match probabilities between team1 and team2, considering cumulative fatigue.
        """
        f1 = min(int(fatigue1), 5)
        f2 = min(int(fatigue2), 5)
        
        if (team1, team2, f1, f2) in self.matchup_cache:
            return self.matchup_cache[(team1, team2, f1, f2)]
            
        # Fallback for teams not in cache
        f1_feat = self.team_features.get(team1, self._precompute_team_features(team1, pd.read_csv("data/results.csv"), None))
        f2_feat = self.team_features.get(team2, self._precompute_team_features(team2, pd.read_csv("data/results.csv"), None))
        
        elo1 = f1_feat['elo'] - f1 * 20.0
        elo2 = f2_feat['elo'] - f2 * 20.0
        elo_diff = elo1 - elo2
        form_diff = f1_feat['form'] - f2_feat['form']
        squad_diff = f1_feat['squad_quality'] - f2_feat['squad_quality']
        
        host_advantage_home = 1 if f1_feat['host_advantage'] == 1 and f2_feat['host_advantage'] == 0 else 0
        host_advantage_away = 1 if f2_feat['host_advantage'] == 1 and f1_feat['host_advantage'] == 0 else 0
        
        row = {
            'elo_diff': elo_diff,
            'home_elo': elo1,
            'away_elo': elo2,
            'home_form': f1_feat['form'],
            'away_form': f2_feat['form'],
            'form_diff': form_diff,
            'home_squad_quality': f1_feat['squad_quality'],
            'away_squad_quality': f2_feat['squad_quality'],
            'squad_quality_diff': squad_diff,
            'home_interim_coach': f1_feat['is_interim'],
            'away_interim_coach': f2_feat['is_interim'],
            'home_advantage': f1_feat['host_advantage'],
            'host_advantage_home': host_advantage_home,
            'host_advantage_away': host_advantage_away,
            
            # New features
            'home_cohesion': f1_feat['cohesion'],
            'away_cohesion': f2_feat['cohesion'],
            'cohesion_diff': f1_feat['cohesion'] - f2_feat['cohesion'],
            'home_peak_age_ratio': f1_feat['peak_age_ratio'],
            'away_peak_age_ratio': f2_feat['peak_age_ratio'],
            'peak_age_ratio_diff': f1_feat['peak_age_ratio'] - f2_feat['peak_age_ratio'],
            'home_veteran_ratio': f1_feat['veteran_ratio'],
            'away_veteran_ratio': f2_feat['veteran_ratio'],
            'veteran_ratio_diff': f1_feat['veteran_ratio'] - f2_feat['veteran_ratio'],
            'home_youth_ratio': f1_feat['youth_ratio'],
            'away_youth_ratio': f2_feat['youth_ratio'],
            'youth_ratio_diff': f1_feat['youth_ratio'] - f2_feat['youth_ratio'],
            'home_tournament_experience': f1_feat['tournament_experience'],
            'away_tournament_experience': f2_feat['tournament_experience'],
            'tournament_experience_diff': f1_feat['tournament_experience'] - f2_feat['tournament_experience'],
            'home_momentum': f1_feat['momentum'],
            'away_momentum': f2_feat['momentum'],
            'momentum_diff': f1_feat['momentum'] - f2_feat['momentum']
        }
        
        df_match = pd.DataFrame([row])[self.feature_cols]
        base_preds = self.base_model.predict_proba(df_match)
        cal_preds = self.calibrator.predict_proba(base_preds)[0]
        cal_preds = np.array(cal_preds, dtype=np.float64)
        cal_preds = cal_preds / cal_preds.sum()
        return cal_preds

    def simulate_match_goals(self, p_home, p_draw, p_away):
        total = p_home + p_draw + p_away
        r = random.random() * total
        
        if r < p_home:  # Home Win
            r_gd = random.random()
            if r_gd < 0.60:
                gd = 1
            elif r_gd < 0.85:
                gd = 2
            elif r_gd < 0.95:
                gd = 3
            else:
                gd = 4
                
            r_lg = random.random()
            if r_lg < 0.55:
                loser_goals = 0
            elif r_lg < 0.90:
                loser_goals = 1
            else:
                loser_goals = 2
            return loser_goals + gd, loser_goals, 0
            
        elif r < p_home + p_draw:  # Draw
            r_g = random.random()
            if r_g < 0.30:
                goals = 0
            elif r_g < 0.80:
                goals = 1
            elif r_g < 0.96:
                goals = 2
            else:
                goals = 3
            return goals, goals, 1
            
        else:  # Away Win
            r_gd = random.random()
            if r_gd < 0.60:
                gd = 1
            elif r_gd < 0.85:
                gd = 2
            elif r_gd < 0.95:
                gd = 3
            else:
                gd = 4
                
            r_lg = random.random()
            if r_lg < 0.55:
                loser_goals = 0
            elif r_lg < 0.90:
                loser_goals = 1
            else:
                loser_goals = 2
            return loser_goals, loser_goals + gd, 2

    def simulate_group_stage(self, track_details=False):
        """Simulates the group stage and returns the qualified teams."""
        group_standings = {}
        all_matches_details = {} if track_details else None
        
        for g_letter, teams in GROUPS_2026.items():
            standings = {t: {'points': 0, 'gd': 0, 'gs': 0, 'elo': self.starting_elos[t], 'team': t} for t in teams}
            group_matches = []
            group_matches_details = []
            
            # Use official FIFA 2026 fixture order for each group.
            # Patterns derived from Kaggle dataset (areezvisram12/fifa-world-cup-2026-match-data-unofficial).
            # Each tuple is (home_team_index, away_team_index) into the group's team list.
            match_indices = GROUP_FIXTURE_PATTERNS[g_letter]
            for idx1, idx2 in match_indices:
                t1, t2 = teams[idx1], teams[idx2]
                probs = self.matchup_cache[(t1, t2, 0, 0)]
                
                # Check for real-world results
                real_score = REAL_WORLD_RESULTS.get((t1, t2))
                if real_score is not None:
                    g1, g2 = real_score
                    outcome = 0 if g1 > g2 else (2 if g2 > g1 else 1)
                else:
                    real_score_rev = REAL_WORLD_RESULTS.get((t2, t1))
                    if real_score_rev is not None:
                        g2_rev, g1_rev = real_score_rev
                        g1, g2 = g1_rev, g2_rev
                        outcome = 0 if g1 > g2 else (2 if g2 > g1 else 1)
                    else:
                        g1, g2, outcome = self.simulate_match_goals(probs[0], probs[1], probs[2])
                
                if outcome == 0:  # t1 wins
                    standings[t1]['points'] += 3
                elif outcome == 2:  # t2 wins
                    standings[t2]['points'] += 3
                else:  # draw
                    standings[t1]['points'] += 1
                    standings[t2]['points'] += 1
                    
                standings[t1]['gd'] += (g1 - g2)
                standings[t1]['gs'] += g1
                
                standings[t2]['gd'] += (g2 - g1)
                standings[t2]['gs'] += g2
                
                group_matches.append((t1, t2, g1, g2))
                
                if track_details:
                    group_matches_details.append({
                        "team1": t1,
                        "team2": t2,
                        "goals1": int(g1),
                        "goals2": int(g2),
                        "probs": [float(p) for p in probs]
                    })
                    
            # Sort standings based on proper FIFA rules: Points, GD, GS, H2H, then Elo
            sorted_teams = sorted(
                standings.values(),
                key=functools.cmp_to_key(lambda x, y: compare_teams(x, y, group_matches)),
                reverse=True
            )
            group_standings[g_letter] = sorted_teams
            if track_details:
                all_matches_details[g_letter] = group_matches_details
            
        if track_details:
            return group_standings, all_matches_details
        return group_standings

    def get_qualified_teams(self, group_standings, track_details=False):
        """Identifies the 32 teams qualifying for the knockout stage."""
        knockout_teams = {}
        third_placed_teams = []
        
        for g_letter, standing in group_standings.items():
            knockout_teams[f"1{g_letter}"] = standing[0]['team']
            knockout_teams[f"2{g_letter}"] = standing[1]['team']
            
            third_team = {k: v for k, v in standing[2].items()}
            third_team['group'] = g_letter
            third_placed_teams.append(third_team)
            
        # Sort third-placed teams (H2H not applicable since they are from different groups)
        sorted_thirds = sorted(
            third_placed_teams,
            key=lambda x: (x['points'], x['gd'], x['gs'], x['elo']),
            reverse=True
        )
        
        best_eight_thirds = sorted_thirds[:8]
        third_places_formatted = [{'team': t['team'], 'group': t['group']} for t in best_eight_thirds]
        
        # Route third place teams to group winners A-H
        winners_a_h = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        third_place_routing = assign_third_places(winners_a_h, third_places_formatted)
        
        for g_winner, tp in third_place_routing.items():
            knockout_teams[f"3rd{g_winner}"] = tp['team']
            
        if track_details:
            third_place_details = {
                "all_third_placed": sorted_thirds,
                "best_eight_thirds": [t['team'] for t in best_eight_thirds],
                "routing": third_place_routing
            }
            return knockout_teams, third_place_details
            
        return knockout_teams

    def simulate_knockout_match(self, team1, team2, fatigue1, fatigue2, match_log=None):
        """Simulates a knockout match to determine who advances and their updated fatigue."""
        f1_c = min(int(fatigue1), 5)
        f2_c = min(int(fatigue2), 5)
        probs = self.matchup_cache[(team1, team2, f1_c, f2_c)]
        p_home, p_draw, p_away = probs[0], probs[1], probs[2]
        
        g1, g2, outcome = self.simulate_match_goals(p_home, p_draw, p_away)
        
        if match_log is not None:
            log_entry = {
                "team1": team1,
                "team2": team2,
                "fatigue1": fatigue1,
                "fatigue2": fatigue2,
                "probs": [float(p) for p in probs],
                "goals1": int(g1),
                "goals2": int(g2),
                "extra_time": False,
                "extra_time_winner": None,
                "penalty_shootout": False,
                "shootout_p1": None,
                "shootout_winner": None,
                "winner": None
            }
        
        if outcome == 0:
            if match_log is not None:
                log_entry["winner"] = team1
                match_log.append(log_entry)
            return team1, fatigue1, fatigue2
        elif outcome == 2:
            if match_log is not None:
                log_entry["winner"] = team2
                match_log.append(log_entry)
            return team2, fatigue1, fatigue2
            
        # Extra Time
        fatigue1 += 1
        fatigue2 += 1
        if match_log is not None:
            log_entry["extra_time"] = True
            log_entry["fatigue1"] = fatigue1
            log_entry["fatigue2"] = fatigue2
        
        et_decided = np.random.rand() < 0.40
        if et_decided:
            p_win_1 = p_home / (p_home + p_away)
            winner = team1 if np.random.rand() < p_win_1 else team2
            if match_log is not None:
                log_entry["extra_time_winner"] = winner
                log_entry["winner"] = winner
                match_log.append(log_entry)
            return winner, fatigue1, fatigue2
            
        # Penalty shootout
        sq1 = self.team_features[team1]['squad_quality']
        sq2 = self.team_features[team2]['squad_quality']
        elo1 = self.starting_elos.get(team1, 1500.0) - fatigue1 * 20.0
        elo2 = self.starting_elos.get(team2, 1500.0) - fatigue2 * 20.0
        
        p_shootout_1 = 0.5 + 0.0008 * (elo1 - elo2) + 0.05 * (sq1 - sq2)
        p_shootout_1 = np.clip(p_shootout_1, 0.15, 0.85)
        
        winner = team1 if np.random.rand() < p_shootout_1 else team2
        if match_log is not None:
            log_entry["penalty_shootout"] = True
            log_entry["shootout_p1"] = float(p_shootout_1)
            log_entry["shootout_winner"] = winner
            log_entry["winner"] = winner
            match_log.append(log_entry)
        return winner, fatigue1, fatigue2

    def simulate_tournament(self, track_details=False):
        """Simulates a single full tournament from group stage to the final."""
        # 1. Group Stage
        if track_details:
            group_standings, group_matches_details = self.simulate_group_stage(track_details=True)
            ko_teams, third_place_details = self.get_qualified_teams(group_standings, track_details=True)
        else:
            group_standings = self.simulate_group_stage()
            ko_teams = self.get_qualified_teams(group_standings)
        
        # Track team fatigue
        fatigue = {t: 0 for t in self.starting_elos.keys()}
        
        # Logs if tracking details
        r32_log = [] if track_details else None
        r16_log = [] if track_details else None
        qf_log = [] if track_details else None
        sf_log = [] if track_details else None
        tp_log = [] if track_details else None
        final_log = [] if track_details else None
        
        # 3. Round of 32
        r32_matches = [
            (ko_teams["1A"], ko_teams.get("3rdA")),
            (ko_teams["1B"], ko_teams.get("3rdB")),
            (ko_teams["1C"], ko_teams.get("3rdC")),
            (ko_teams["1D"], ko_teams.get("3rdD")),
            (ko_teams["1E"], ko_teams.get("3rdE")),
            (ko_teams["1F"], ko_teams.get("3rdF")),
            (ko_teams["1G"], ko_teams.get("3rdG")),
            (ko_teams["1H"], ko_teams.get("3rdH")),
            (ko_teams["1I"], ko_teams["2J"]),
            (ko_teams["1J"], ko_teams["2I"]),
            (ko_teams["1K"], ko_teams["2L"]),
            (ko_teams["1L"], ko_teams["2K"]),
            (ko_teams["2A"], ko_teams["2B"]),
            (ko_teams["2C"], ko_teams["2D"]),
            (ko_teams["2E"], ko_teams["2F"]),
            (ko_teams["2G"], ko_teams["2H"]),
        ]
        
        r16_teams = []
        for t1, t2 in r32_matches:
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2], match_log=r32_log)
            fatigue[t1] = f1
            fatigue[t2] = f2
            r16_teams.append(w)
            
        # 4. Round of 16
        r16_matches = [
            (r16_teams[0], r16_teams[8]),
            (r16_teams[1], r16_teams[9]),
            (r16_teams[2], r16_teams[10]),
            (r16_teams[3], r16_teams[11]),
            (r16_teams[4], r16_teams[12]),
            (r16_teams[5], r16_teams[13]),
            (r16_teams[6], r16_teams[14]),
            (r16_teams[7], r16_teams[15]),
        ]
        
        qf_teams = []
        for t1, t2 in r16_matches:
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2], match_log=r16_log)
            fatigue[t1] = f1
            fatigue[t2] = f2
            qf_teams.append(w)
            
        # 5. Quarterfinals
        qf_matches = [
            (qf_teams[0], qf_teams[4]),
            (qf_teams[1], qf_teams[5]),
            (qf_teams[2], qf_teams[6]),
            (qf_teams[3], qf_teams[7]),
        ]
        
        sf_teams = []
        for t1, t2 in qf_matches:
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2], match_log=qf_log)
            fatigue[t1] = f1
            fatigue[t2] = f2
            sf_teams.append(w)
            
        # 6. Semifinals
        sf_matches = [
            (sf_teams[0], sf_teams[2]),
            (sf_teams[1], sf_teams[3]),
        ]
        
        final_teams = []
        third_place_teams = []
        for t1, t2 in sf_matches:
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2], match_log=sf_log)
            fatigue[t1] = f1
            fatigue[t2] = f2
            final_teams.append(w)
            
            loser = t1 if w == t2 else t2
            third_place_teams.append(loser)
            
        # 7. Third Place Match
        tp_winner, _, _ = self.simulate_knockout_match(
            third_place_teams[0], third_place_teams[1],
            fatigue[third_place_teams[0]], fatigue[third_place_teams[1]],
            match_log=tp_log
        )
        
        # 8. Final
        champion, _, _ = self.simulate_knockout_match(
            final_teams[0], final_teams[1],
            fatigue[final_teams[0]], fatigue[final_teams[1]],
            match_log=final_log
        )
        
        result = {
            "group_standings": group_standings,
            "r32_teams": list(ko_teams.values()),
            "r16_teams": r16_teams,
            "qf_teams": qf_teams,
            "sf_teams": sf_teams,
            "third_place": tp_winner,
            "runner_up": final_teams[0] if champion == final_teams[1] else final_teams[1],
            "champion": champion
        }
        
        if track_details:
            result["details"] = {
                "group_matches": group_matches_details,
                "group_standings_clean": {g: [{k: v for k, v in t.items()} for t in standing] for g, standing in group_standings.items()},
                "third_place_routing": third_place_details,
                "r32_matches": r32_log,
                "r16_matches": r16_log,
                "qf_matches": qf_log,
                "sf_matches": sf_log,
                "third_place_match": tp_log,
                "final_match": final_log
            }
            
        return result

    def run_monte_carlo(self, num_simulations=1000, num_batches=10):
        """Runs the simulation in batches and returns mean probabilities and 95% confidence intervals."""
        runs_per_batch = max(1, num_simulations // num_batches)
        stages = ["group_stage_exit", "r32_exit", "r16_exit", "qf_exit", "sf_exit", "third_place", "runner_up", "champion"]
        
        batch_results = {t: {s: [] for s in stages} for t in self.starting_elos.keys()}
        
        for b in range(num_batches):
            counts = {t: {s: 0 for s in stages} for t in self.starting_elos.keys()}
            for _ in range(runs_per_batch):
                res = self.simulate_tournament()
                
                champ = res["champion"]
                runner = res["runner_up"]
                third = res["third_place"]
                
                counts[champ]["champion"] += 1
                counts[runner]["runner_up"] += 1
                counts[third]["third_place"] += 1
                
                for t in self.starting_elos.keys():
                    if t == champ or t == runner or t == third:
                        continue
                    if t in res["sf_teams"]:
                        counts[t]["sf_exit"] += 1
                    elif t in res["qf_teams"]:
                        counts[t]["qf_exit"] += 1
                    elif t in res["r16_teams"]:
                        counts[t]["r16_exit"] += 1
                    elif t in res["r32_teams"]:
                        counts[t]["r32_exit"] += 1
                    else:
                        counts[t]["group_stage_exit"] += 1
                        
            for t in self.starting_elos.keys():
                for s in stages:
                    batch_results[t][s].append(counts[t][s] / runs_per_batch)
                    
        summary = {}
        for t in self.starting_elos.keys():
            summary[t] = {}
            for s in stages:
                vals = np.array(batch_results[t][s])
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                se = std_val / math.sqrt(num_batches)
                ci_lower = max(0.0, mean_val - 1.96 * se)
                ci_upper = min(1.0, mean_val + 1.96 * se)
                
                summary[t][s] = {
                    "mean": mean_val,
                    "std": std_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper
                }
                
        return summary
