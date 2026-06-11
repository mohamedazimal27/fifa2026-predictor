import os
import pickle
import json
import random
import numpy as np
import pandas as pd
from src.data_pipeline.curated_lookup import CuratedLookup
from src.data_pipeline.data_loader import parse_elo_tsv
from src.third_place_router import assign_third_places

# 2026 World Cup Groups and Teams
GROUPS_2026 = {
    'A': ['United_States', 'Colombia', 'Morocco', 'Australia'],
    'B': ['Canada', 'Italy', 'Japan', 'Senegal'],
    'C': ['Mexico', 'Uruguay', 'South_Korea', 'Poland'],
    'D': ['Argentina', 'Denmark', 'Ecuador', 'Saudi_Arabia'],
    'E': ['Brazil', 'Switzerland', 'Cameroon', 'Iran'],
    'F': ['France', 'Croatia', 'Chile', 'Tunisia'],
    'G': ['England', 'Peru', 'Nigeria', 'Qatar'],
    'H': ['Spain', 'Sweden', 'Egypt', 'Costa_Rica'],
    'I': ['Germany', 'Ukraine', 'Algeria', 'Iraq'],
    'J': ['Portugal', 'Wales', 'Ghana', 'Panama'],
    'K': ['Netherlands', 'Austria', 'Mali', 'Jamaica'],
    'L': ['Belgium', 'Turkey', 'Ivory_Coast', 'New_Zealand']
}

class TournamentSimulator:
    def __init__(self, model_path="models/fifa_model.pkl", curated_path="data/curated_teams.json", elo_dir="data/elo", canonical_path="data/canonical_teams.json"):
        self.curated_path = curated_path
        self.elo_dir = elo_dir
        
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
            
        # Get starting Elo ratings for all 2026 teams
        self.starting_elos = {}
        for group, teams in GROUPS_2026.items():
            for team in teams:
                self.starting_elos[team] = self._get_latest_elo(team)
                
        # Pre-cache team features to avoid looking them up repeatedly
        self.team_features = {}
        for group, teams in GROUPS_2026.items():
            for team in teams:
                self.team_features[team] = self._get_team_features_for_2026(team)
                
        # Pre-compute match probabilities cache for all 48 teams
        self.matchup_cache = {}
        self._precompute_matchups()

    def _get_latest_elo(self, team_name):
        if team_name not in self.canonical_mapping:
            return 1500.0
        code = self.canonical_mapping[team_name]['code']
        elo_file = self.canonical_mapping[team_name]['elo_file']
        tsv_path = os.path.join(self.elo_dir, elo_file)
        if not os.path.exists(tsv_path):
            return 1500.0
        df_elo = parse_elo_tsv(tsv_path)
        if df_elo.empty:
            return 1500.0
        # Get the last row
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

    def _get_team_features_for_2026(self, team, date_str="2026-06-11"):
        # ELO
        elo = self.starting_elos.get(team, 1500.0)
        
        # Form: 2026 World Cup has starting forms. Let's default to 0.0
        # since it's the start of the tournament.
        form = 0.0
        
        # Curated features (squad quality, is_interim)
        curated = self.lookup_system.lookup(team, date_str)
        squad_quality = curated['squad_quality']
        is_interim = 1 if curated['is_interim'] else 0
        
        # Host Advantage: 1 if USA, Mexico, Canada
        host_adv = 1 if team in ['United_States', 'Mexico', 'Canada'] else 0
        
        return {
            'elo': elo,
            'form': form,
            'squad_quality': squad_quality,
            'is_interim': is_interim,
            'host_advantage': host_adv
        }

    def _precompute_matchups(self):
        """Pre-computes and caches probabilities for all 48 teams in pairwise matchups."""
        teams = list(self.starting_elos.keys())
        n_teams = len(teams)
        
        # We'll build a batch of all possible matchups to predict in one go
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
                        # Apply fatigue Elo penalty (20 Elo points per unit of fatigue)
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
                            'host_advantage_away': host_advantage_away
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

    def predict_match(self, team1, team2, fatigue1=0, fatigue2=0):
        """
        Predicts match probabilities between team1 and team2, considering cumulative fatigue.
        """
        f1 = min(int(fatigue1), 5)
        f2 = min(int(fatigue2), 5)
        
        # Read from pre-computed cache
        if (team1, team2, f1, f2) in self.matchup_cache:
            return self.matchup_cache[(team1, team2, f1, f2)]
            
        # Fallback (should never be reached unless team name is not in the cached 48 teams)
        f1_feat = self._get_team_features_for_2026(team1)
        f2_feat = self._get_team_features_for_2026(team2)
        
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
            'host_advantage_away': host_advantage_away
        }
        
        df_match = pd.DataFrame([row])[self.feature_cols]
        base_preds = self.base_model.predict_proba(df_match)
        cal_preds = self.calibrator.predict_proba(base_preds)[0]
        cal_preds = np.array(cal_preds, dtype=np.float64)
        cal_preds = cal_preds / cal_preds.sum()
        return cal_preds

    def simulate_match_goals(self, p_home, p_draw, p_away):
        """
        Simulates match goals based on match outcome probabilities.
        """
        total = p_home + p_draw + p_away
        r = random.random() * total
        
        if r < p_home:  # Home Win
            # gd choice: [1, 2, 3, 4] with p=[0.60, 0.25, 0.10, 0.05]
            r_gd = random.random()
            if r_gd < 0.60:
                gd = 1
            elif r_gd < 0.85:
                gd = 2
            elif r_gd < 0.95:
                gd = 3
            else:
                gd = 4
                
            # loser_goals choice: [0, 1, 2] with p=[0.55, 0.35, 0.10]
            r_lg = random.random()
            if r_lg < 0.55:
                loser_goals = 0
            elif r_lg < 0.90:
                loser_goals = 1
            else:
                loser_goals = 2
                
            return loser_goals + gd, loser_goals, 0
            
        elif r < p_home + p_draw:  # Draw
            # goals choice: [0, 1, 2, 3] with p=[0.30, 0.50, 0.16, 0.04]
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
            # gd choice: [1, 2, 3, 4] with p=[0.60, 0.25, 0.10, 0.05]
            r_gd = random.random()
            if r_gd < 0.60:
                gd = 1
            elif r_gd < 0.85:
                gd = 2
            elif r_gd < 0.95:
                gd = 3
            else:
                gd = 4
                
            # loser_goals choice: [0, 1, 2] with p=[0.55, 0.35, 0.10]
            r_lg = random.random()
            if r_lg < 0.55:
                loser_goals = 0
            elif r_lg < 0.90:
                loser_goals = 1
            else:
                loser_goals = 2
                
            return loser_goals, loser_goals + gd, 2

    def simulate_group_stage(self):
        """Simulates the group stage and returns the qualified teams."""
        group_standings = {}
        
        for g_letter, teams in GROUPS_2026.items():
            standings = {t: {'points': 0, 'gd': 0, 'gs': 0, 'elo': self.starting_elos[t], 'team': t} for t in teams}
            
            # Play round robin (6 matches)
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    probs = self.matchup_cache[(t1, t2, 0, 0)]
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
                    
            # Sort standings based on FIFA rules: Points, GD, GS, then Elo
            sorted_teams = sorted(
                standings.values(),
                key=lambda x: (x['points'], x['gd'], x['gs'], x['elo']),
                reverse=True
            )
            group_standings[g_letter] = sorted_teams
            
        return group_standings

    def get_qualified_teams(self, group_standings):
        """Identifies the 32 teams qualifying for the knockout stage."""
        knockout_teams = {}
        third_placed_teams = []
        
        for g_letter, standing in group_standings.items():
            # Top 2 teams qualify directly
            knockout_teams[f"1{g_letter}"] = standing[0]['team']
            knockout_teams[f"2{g_letter}"] = standing[1]['team']
            
            # Keep track of 3rd place team
            third_team = standing[2]
            third_team['group'] = g_letter
            third_placed_teams.append(third_team)
            
        # Sort third-placed teams to find the best 8
        sorted_thirds = sorted(
            third_placed_teams,
            key=lambda x: (x['points'], x['gd'], x['gs'], x['elo']),
            reverse=True
        )
        
        best_eight_thirds = sorted_thirds[:8]
        
        # Format for third place router
        third_places_formatted = [{'team': t['team'], 'group': t['group']} for t in best_eight_thirds]
        
        # Route third place teams to group winners A-H
        winners_a_h = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        third_place_routing = assign_third_places(winners_a_h, third_places_formatted)
        
        for g_winner, tp in third_place_routing.items():
            knockout_teams[f"3rd{g_winner}"] = tp['team']
            
        return knockout_teams

    def simulate_knockout_match(self, team1, team2, fatigue1, fatigue2):
        """Simulates a knockout match to determine who advances and their updated fatigue."""
        f1_c = min(int(fatigue1), 5)
        f2_c = min(int(fatigue2), 5)
        probs = self.matchup_cache[(team1, team2, f1_c, f2_c)]
        p_home, p_draw, p_away = probs[0], probs[1], probs[2]
        
        # Regular time goals
        g1, g2, outcome = self.simulate_match_goals(p_home, p_draw, p_away)
        
        if outcome == 0:
            return team1, fatigue1, fatigue2
        elif outcome == 2:
            return team2, fatigue1, fatigue2
            
        # Draw in regular time: Go to extra time (increases fatigue)
        fatigue1 += 1
        fatigue2 += 1
        
        # Simulate extra-time decision (40% chance of deciding in ET, 60% chance of penalties)
        et_decided = np.random.rand() < 0.40
        if et_decided:
            # Decide based on relative strength
            p_win_1 = p_home / (p_home + p_away)
            winner = team1 if np.random.rand() < p_win_1 else team2
            return winner, fatigue1, fatigue2
            
        # Penalty shootout
        # Shootout model
        sq1 = self.team_features[team1]['squad_quality']
        sq2 = self.team_features[team2]['squad_quality']
        elo1 = self.starting_elos.get(team1, 1500.0) - fatigue1 * 20.0
        elo2 = self.starting_elos.get(team2, 1500.0) - fatigue2 * 20.0
        
        p_shootout_1 = 0.5 + 0.0008 * (elo1 - elo2) + 0.05 * (sq1 - sq2)
        p_shootout_1 = np.clip(p_shootout_1, 0.15, 0.85)
        
        winner = team1 if np.random.rand() < p_shootout_1 else team2
        return winner, fatigue1, fatigue2

    def simulate_tournament(self):
        """Simulates a single full tournament from group stage to the final."""
        # 1. Group Stage
        group_standings = self.simulate_group_stage()
        
        # 2. Qualified Teams & Routing
        ko_teams = self.get_qualified_teams(group_standings)
        
        # Track team fatigue
        fatigue = {t: 0 for t in self.starting_elos.keys()}
        
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
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2])
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
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2])
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
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2])
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
            w, f1, f2 = self.simulate_knockout_match(t1, t2, fatigue[t1], fatigue[t2])
            fatigue[t1] = f1
            fatigue[t2] = f2
            final_teams.append(w)
            
            loser = t1 if w == t2 else t2
            third_place_teams.append(loser)
            
        # 7. Third Place Match
        tp_winner, _, _ = self.simulate_knockout_match(
            third_place_teams[0], third_place_teams[1],
            fatigue[third_place_teams[0]], fatigue[third_place_teams[1]]
        )
        
        # 8. Final
        champion, _, _ = self.simulate_knockout_match(
            final_teams[0], final_teams[1],
            fatigue[final_teams[0]], fatigue[final_teams[1]]
        )
        
        return {
            "group_standings": group_standings,
            "r32_teams": list(ko_teams.values()),
            "r16_teams": r16_teams,
            "qf_teams": qf_teams,
            "sf_teams": sf_teams,
            "third_place": tp_winner,
            "runner_up": final_teams[0] if champion == final_teams[1] else final_teams[1],
            "champion": champion
        }

    def run_monte_carlo(self, num_simulations=1000):
        """Runs the simulation multiple times and aggregates probability statistics."""
        stats = {t: {
            "group_stage_exit": 0,
            "r32_exit": 0,
            "r16_exit": 0,
            "qf_exit": 0,
            "sf_exit": 0,
            "third_place": 0,
            "runner_up": 0,
            "champion": 0
        } for t in self.starting_elos.keys()}
        
        for _ in range(num_simulations):
            res = self.simulate_tournament()
            
            # Extract progression
            champ = res["champion"]
            runner = res["runner_up"]
            third = res["third_place"]
            
            stats[champ]["champion"] += 1
            stats[runner]["runner_up"] += 1
            stats[third]["third_place"] += 1
            
            # Map exits
            for t in self.starting_elos.keys():
                if t == champ or t == runner or t == third:
                    continue
                if t in res["sf_teams"]:
                    stats[t]["sf_exit"] += 1
                elif t in res["qf_teams"]:
                    stats[t]["qf_exit"] += 1
                elif t in res["r16_teams"]:
                    stats[t]["r16_exit"] += 1
                elif t in res["r32_teams"]:
                    stats[t]["r32_exit"] += 1
                else:
                    stats[t]["group_stage_exit"] += 1
                    
        # Normalize to probabilities
        probs = {}
        for team, counts in stats.items():
            probs[team] = {k: v / num_simulations for k, v in counts.items()}
            
        return probs
