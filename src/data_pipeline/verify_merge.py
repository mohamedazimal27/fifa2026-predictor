import os
import pandas as pd
from src.data_pipeline.data_loader import merge_elo_and_results

def main():
    results_path = "data/results.csv"
    canonical_path = "data/canonical_teams.json"
    elo_dir = "data/elo"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    df = merge_elo_and_results(results_path, canonical_path, elo_dir)
    
    # Save merged data to data/merged_results.csv
    merged_output_path = "data/merged_results.csv"
    df.to_csv(merged_output_path, index=False)
    print(f"Saved merged results to {merged_output_path}")
    
    # Filter for 2018 World Cup
    wc_2018 = df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == 2018)]
    print(f"\nTotal 2018 World Cup matches found: {len(wc_2018)}")
    
    # Print sample matches
    sample_matches = [
        ("Russia", "Saudi Arabia"),
        ("Portugal", "Spain"),
        ("France", "Australia"),
        ("Argentina", "Iceland"),
        ("Brazil", "Switzerland"),
        ("Germany", "Mexico")
    ]
    
    print("\n--- Verifying Sample 2018 World Cup Matches ---")
    for home, away in sample_matches:
        match = wc_2018[((wc_2018['home_team'] == home) & (wc_2018['away_team'] == away)) |
                        ((wc_2018['home_team'] == away) & (wc_2018['away_team'] == home))]
        if not match.empty:
            row = match.iloc[0]
            print(f"Date: {row['date'].strftime('%Y-%m-%d')}")
            print(f"Match: {row['home_team']} vs {row['away_team']}")
            print(f"Score: {row['home_score']} - {row['away_score']}")
            print(f"Elo Before: {row['home_team']} = {row['home_elo_before']:.1f}, {row['away_team']} = {row['away_elo_before']:.1f}")
            print("-" * 40)
        else:
            print(f"Match {home} vs {away} not found in 2018 World Cup.")

if __name__ == "__main__":
    main()
