import os
import pytest
from hypothesis import given, strategies as st
from src.simulator import TournamentSimulator, compare_teams

# Instantiate simulator once for use across tests to avoid repeated heavy model loading
sim = TournamentSimulator()

def test_tournament_simulator():
    # Run a single simulation
    res = sim.simulate_tournament()
    
    # Check champion is a valid team
    assert res["champion"] in sim.starting_elos
    assert res["runner_up"] in sim.starting_elos
    assert res["third_place"] in sim.starting_elos
    
    # Check that winner/runner/third are distinct
    assert len({res["champion"], res["runner_up"], res["third_place"]}) == 3
    
    # Check knockout stage dimensions
    assert len(res["r32_teams"]) == 32
    assert len(res["r16_teams"]) == 16
    assert len(res["qf_teams"]) == 8
    assert len(res["sf_teams"]) == 4

def test_monte_carlo():
    # Run a small batch of simulations
    probs = sim.run_monte_carlo(num_simulations=10)
    
    # Verify we get probabilities for all 48 teams
    assert len(probs) == 48
    
    # Check that probabilities sum to approximately 1 for champion/runner_up/third_place
    sum_champs = sum(p["champion"]["mean"] for p in probs.values())
    assert abs(sum_champs - 1.0) < 1e-9
    
    sum_runners = sum(p["runner_up"]["mean"] for p in probs.values())
    assert abs(sum_runners - 1.0) < 1e-9
    
    sum_thirds = sum(p["third_place"]["mean"] for p in probs.values())
    assert abs(sum_thirds - 1.0) < 1e-9


# --- Hypothesis Property-Based Tests ---

team_name_strategy = st.sampled_from(list(sim.starting_elos.keys()))

@given(
    team_a=team_name_strategy,
    team_b=team_name_strategy,
    fatigue_a=st.integers(min_value=0, max_value=10),
    fatigue_b=st.integers(min_value=0, max_value=10)
)
def test_matchup_probabilities_properties(team_a, team_b, fatigue_a, fatigue_b):
    if team_a == team_b:
        return
    probs = sim.predict_match(team_a, team_b, fatigue_a, fatigue_b)
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Calibrated probabilities should sum to 1.0
    assert abs(sum(probs) - 1.0) < 1e-7


team_dict_strategy = st.fixed_dictionaries({
    'team': st.text(min_size=1, max_size=15),
    'points': st.integers(min_value=0, max_value=9),
    'gd': st.integers(min_value=-15, max_value=15),
    'gs': st.integers(min_value=0, max_value=30),
    'elo': st.floats(min_value=500.0, max_value=2500.0, allow_nan=False, allow_infinity=False)
})

match_strategy = st.lists(
    st.tuples(
        st.text(min_size=1, max_size=15),
        st.text(min_size=1, max_size=15),
        st.integers(min_value=0, max_value=10),
        st.integers(min_value=0, max_value=10)
    ),
    max_size=6
)

@given(
    a=team_dict_strategy,
    b=team_dict_strategy,
    group_matches=match_strategy
)
def test_compare_teams_anti_symmetry(a, b, group_matches):
    if a['team'] == b['team']:
        return
    res_ab = compare_teams(a, b, group_matches)
    res_ba = compare_teams(b, a, group_matches)
    
    if res_ab > 0:
        assert res_ba < 0
    elif res_ab < 0:
        assert res_ba > 0
    else:
        assert res_ba == 0


def test_tournament_simulator_detail_tracking():
    # Run simulation with detail tracking
    res = sim.simulate_tournament(track_details=True)
    
    # Check that the standard keys exist
    assert res["champion"] in sim.starting_elos
    assert res["runner_up"] in sim.starting_elos
    assert res["third_place"] in sim.starting_elos
    
    # Check that 'details' key exists and has the correct keys
    assert "details" in res
    details = res["details"]
    
    expected_keys = [
        "group_matches",
        "group_standings_clean",
        "third_place_routing",
        "r32_matches",
        "r16_matches",
        "qf_matches",
        "sf_matches",
        "third_place_match",
        "final_match"
    ]
    for key in expected_keys:
        assert key in details
        
    # Check group stage match count (12 groups * 6 matches = 72)
    group_matches = details["group_matches"]
    assert len(group_matches) == 12
    total_group_matches = sum(len(matches) for matches in group_matches.values())
    assert total_group_matches == 72
    
    # Check group standings clean count (12 groups * 4 teams = 48)
    standings_clean = details["group_standings_clean"]
    assert len(standings_clean) == 12
    for group, teams in standings_clean.items():
        assert len(teams) == 4
        
    # Check third place routing
    tp_routing = details["third_place_routing"]
    assert "all_third_placed" in tp_routing
    assert len(tp_routing["all_third_placed"]) == 12
    assert "best_eight_thirds" in tp_routing
    assert len(tp_routing["best_eight_thirds"]) == 8
    assert "routing" in tp_routing
    assert len(tp_routing["routing"]) == 8
    
    # Check knockout matches count
    assert len(details["r32_matches"]) == 16
    assert len(details["r16_matches"]) == 8
    assert len(details["qf_matches"]) == 4
    assert len(details["sf_matches"]) == 2
    assert len(details["third_place_match"]) == 1
    assert len(details["final_match"]) == 1
    
    # Check match structure for a knockout match
    first_r32 = details["r32_matches"][0]
    assert "team1" in first_r32
    assert "team2" in first_r32
    assert "probs" in first_r32
    assert len(first_r32["probs"]) == 3
    assert "goals1" in first_r32
    assert "goals2" in first_r32
    assert "extra_time" in first_r32
    assert "winner" in first_r32


def test_real_world_seeding_and_indices_order():
    # Simulate a single group stage run
    group_standings, all_matches_details = sim.simulate_group_stage(track_details=True)
    
    # 1. Verify the match_indices order for Group A
    # Mexico vs South_Africa (Match 1)
    # South_Korea vs Czechia (Match 2)
    # Czechia vs South_Africa (Match 3)
    # Mexico vs South_Korea (Match 4)
    # South_Africa vs South_Korea (Match 5)
    # Czechia vs Mexico (Match 6)
    group_a_matches = all_matches_details["A"]
    assert len(group_a_matches) == 6
    
    # Matchday 1
    assert group_a_matches[0]["team1"] == "Mexico" and group_a_matches[0]["team2"] == "South_Africa"
    assert group_a_matches[1]["team1"] == "South_Korea" and group_a_matches[1]["team2"] == "Czechia"
    # Matchday 2
    assert group_a_matches[2]["team1"] == "Czechia" and group_a_matches[2]["team2"] == "South_Africa"
    assert group_a_matches[3]["team1"] == "Mexico" and group_a_matches[3]["team2"] == "South_Korea"
    # Matchday 3
    assert group_a_matches[4]["team1"] == "South_Africa" and group_a_matches[4]["team2"] == "South_Korea"
    assert group_a_matches[5]["team1"] == "Czechia" and group_a_matches[5]["team2"] == "Mexico"
    
    # 2. Verify that real-world seeded results are enforced correctly
    # Mexico 2-0 South Africa
    assert group_a_matches[0]["goals1"] == 2
    assert group_a_matches[0]["goals2"] == 0
    # South Korea 2-1 Czechia
    assert group_a_matches[1]["goals1"] == 2
    assert group_a_matches[1]["goals2"] == 1
