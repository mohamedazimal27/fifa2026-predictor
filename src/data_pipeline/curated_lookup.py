import os
import json
import pandas as pd

class CuratedLookup:
    def __init__(self, curated_path="data/curated_teams.json"):
        self.curated_path = curated_path
        self.data = {}
        if os.path.exists(curated_path):
            with open(curated_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                
    def lookup(self, team_name: str, match_date) -> dict:
        """
        Looks up the coach, interim status, and squad quality for a team at a specific date.
        match_date can be a string, datetime, or pandas Timestamp.
        """
        if pd.isna(match_date):
            return {
                "coach": "Unknown",
                "is_interim": False,
                "squad_quality": 1.0
            }
            
        dt = pd.to_datetime(match_date)
        
        result = {
            "coach": "Unknown",
            "is_interim": False,
            "squad_quality": 1.0
        }
        
        if team_name not in self.data:
            return result
            
        team_info = self.data[team_name]
        
        # 1. Lookup Coach
        found_coach = "Unknown"
        is_interim = False
        for coach in team_info.get("coaches", []):
            start_dt = pd.to_datetime(coach["start"])
            end_dt = pd.to_datetime(coach["end"])
            if start_dt <= dt <= end_dt:
                found_coach = coach["name"]
                tenure_days = (end_dt - start_dt).days
                is_interim = (tenure_days < 180)
                break
                
        # 2. Lookup Squad Quality
        squad_quality = 1.0
        for sq in team_info.get("squad_quality", []):
            start_dt = pd.to_datetime(sq["start"])
            end_dt = pd.to_datetime(sq["end"])
            if start_dt <= dt <= end_dt:
                squad_quality = float(sq["score"])
                break
                
        return {
            "coach": found_coach,
            "is_interim": is_interim,
            "squad_quality": squad_quality
        }
