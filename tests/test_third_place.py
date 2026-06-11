import itertools
from src.third_place_router import assign_third_places

def test_all_third_place_combinations():
    groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    winners = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Generate all combinations of 8 groups qualifying as 3rd place
    combos = list(itertools.combinations(groups, 8))
    assert len(combos) == 495
    
    for combo in combos:
        third_places = [{'team': f"Team_{g}", 'group': g} for g in combo]
        mapping = assign_third_places(winners, third_places)
        
        # Check that we got exactly 8 assignments
        assert len(mapping) == 8
        
        # Check that each winner plays a team not from their own group
        for w, tp in mapping.items():
            assert w != tp['group'], f"Same-group clash: Winner {w} plays team from Group {tp['group']}"
            
        # Check that all 8 third-placed teams were assigned exactly once
        assigned_groups = [tp['group'] for tp in mapping.values()]
        assert sorted(assigned_groups) == sorted(combo)
