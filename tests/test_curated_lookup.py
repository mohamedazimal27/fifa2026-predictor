import os
import json
import tempfile
import pandas as pd
from src.data_pipeline.curated_lookup import CuratedLookup

def test_curated_lookup():
    # Mock curated data
    mock_data = {
        "Spain": {
            "coaches": [
                {"name": "Fernando Hierro", "start": "2018-06-13", "end": "2018-07-08"}, # 25 days (interim)
                {"name": "Luis Enrique", "start": "2018-07-09", "end": "2019-06-19"} # 345 days (not interim)
            ],
            "squad_quality": [
                {"start": "2018-01-01", "end": "2018-12-31", "score": 1.20}
            ]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(mock_data, f)
        temp_path = f.name
        
    try:
        lookup_system = CuratedLookup(temp_path)
        
        # Test interim coach during 2018 World Cup
        res = lookup_system.lookup("Spain", "2018-06-20")
        assert res["coach"] == "Fernando Hierro"
        assert res["is_interim"] is True
        assert res["squad_quality"] == 1.20
        
        # Test non-interim coach later
        res = lookup_system.lookup("Spain", "2018-10-15")
        assert res["coach"] == "Luis Enrique"
        assert res["is_interim"] is False
        assert res["squad_quality"] == 1.20
        
        # Test unknown team fallback
        res = lookup_system.lookup("Germany", "2018-06-20")
        assert res["coach"] == "Unknown"
        assert res["is_interim"] is False
        assert res["squad_quality"] == 1.0
        
        # Test out of range date fallback
        res = lookup_system.lookup("Spain", "2020-01-01")
        assert res["coach"] == "Unknown"
        assert res["is_interim"] is False
        assert res["squad_quality"] == 1.0
        
    finally:
        os.unlink(temp_path)
