def assign_third_places(winners, third_places):
    """
    Deterministically routes 8 best third-placed teams to 8 group winners (A-H).
    
    Parameters:
    -----------
    winners : list of str
        The group letters of the winners who play against third-placed teams.
        Typically: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    third_places : list of dict
        The 8 qualified third-placed teams, each with 'team' and 'group' (e.g. {'team': 'Morocco', 'group': 'A'}).
        
    Returns:
    --------
    dict
        A mapping from winner group letter to the third-placed team dict.
        Example: {'A': {'team': 'Morocco', 'group': 'C'}, ...}
    """
    # Sort both winners and third-places to ensure deterministic matching
    winners_sorted = sorted(winners)
    # Sort third_places by their source group letter to make backtracking deterministic
    third_places_sorted = sorted(third_places, key=lambda x: x['group'])
    
    assignment = {}
    used_indices = set()
    
    def backtrack(winner_idx):
        if winner_idx == len(winners_sorted):
            return True
            
        w_group = winners_sorted[winner_idx]
        
        for tp_idx, tp in enumerate(third_places_sorted):
            if tp_idx not in used_indices:
                # Rule: Third-placed team cannot play the winner of their own group
                if tp['group'] != w_group:
                    assignment[w_group] = tp
                    used_indices.add(tp_idx)
                    
                    if backtrack(winner_idx + 1):
                        return True
                        
                    # Undo
                    used_indices.remove(tp_idx)
                    del assignment[w_group]
                    
        return False
        
    success = backtrack(0)
    if success:
        return assignment
    else:
        # Fallback in case no valid bipartite matching is found (mathematically virtually impossible with 8 out of 12 groups)
        # Simply pair them sequentially and ignore the same-group rule as a last resort
        return {w: tp for w, tp in zip(winners_sorted, third_places_sorted)}
