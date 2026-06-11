import os
import sys

# Add the project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import TournamentSimulator

def run_verification(num_simulations=10000):
    print(f"Initializing TournamentSimulator for {num_simulations} runs of routing verification...")
    sim = TournamentSimulator()
    
    clashes = 0
    incomplete_placements = 0
    double_fills = 0
    invalid_r32_counts = 0
    
    for i in range(num_simulations):
        # 1. Group Stage
        group_standings = sim.simulate_group_stage()
        
        # 2. Qualified Teams & Routing
        ko_teams = sim.get_qualified_teams(group_standings)
        
        # Verify exactly 32 teams are qualified
        if len(ko_teams) != 32:
            invalid_r32_counts += 1
            
        # Verify no team is double-filled/assigned to multiple slots
        teams_list = list(ko_teams.values())
        if len(set(teams_list)) != 32:
            double_fills += 1
            
        # Verify 3rd place assignments
        winners_a_h = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        # We need to trace where the third placed team came from
        # Let's map team name -> group it came from
        team_to_group = {}
        for g_letter, standing in group_standings.items():
            for entry in standing:
                team_to_group[entry['team']] = g_letter
                
        # Check matchups for winners A-H
        for w_group in winners_a_h:
            winner_team = ko_teams.get(f"1{w_group}")
            third_place_team = ko_teams.get(f"3rd{w_group}")
            
            if not winner_team or not third_place_team:
                incomplete_placements += 1
                continue
                
            tp_group = team_to_group.get(third_place_team)
            
            # Check for same group clash
            if tp_group == w_group:
                clashes += 1
                print(f"Clash in sim {i}: Winner of Group {w_group} ({winner_team}) plays 3rd place from Group {tp_group} ({third_place_team})")

    print("\n--- Router Verification Results ---")
    print(f"Total Simulations Run:     {num_simulations}")
    print(f"Same-Group Rematch Clashes: {clashes}")
    print(f"Incomplete Placements:     {incomplete_placements}")
    print(f"Double-Fill Bracket Errors: {double_fills}")
    print(f"Invalid R32 Team Counts:    {invalid_r32_counts}")
    
    if clashes == 0 and incomplete_placements == 0 and double_fills == 0 and invalid_r32_counts == 0:
        print("SUCCESS: Bracket routing is 100% correct, clash-free, and complete!")
        return True
    else:
        print("FAILURE: Router routing contains logic errors.")
        return False

if __name__ == "__main__":
    run_verification()
