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
