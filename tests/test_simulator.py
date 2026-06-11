import os
from src.simulator import TournamentSimulator

def test_tournament_simulator():
    # Instantiate simulator (will load model and pre-compute)
    sim = TournamentSimulator()
    
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
    sim = TournamentSimulator()
    
    # Run a small batch of simulations
    probs = sim.run_monte_carlo(num_simulations=10)
    
    # Verify we get probabilities for all 48 teams
    assert len(probs) == 48
    
    # Check that probabilities sum to approximately 1 for champion/runner_up/third_place
    sum_champs = sum(p["champion"] for p in probs.values())
    assert abs(sum_champs - 1.0) < 1e-9
    
    sum_runners = sum(p["runner_up"] for p in probs.values())
    assert abs(sum_runners - 1.0) < 1e-9
    
    sum_thirds = sum(p["third_place"] for p in probs.values())
    assert abs(sum_thirds - 1.0) < 1e-9
