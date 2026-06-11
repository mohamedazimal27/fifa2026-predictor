import os
import json
import tempfile
import pytest
import pandas as pd
from src.data_pipeline.data_loader import parse_elo_tsv, merge_elo_and_results

def test_parse_elo_tsv_tab():
    # Mock a tab-separated TSV file without headers
    tsv_content = (
        "1914\t09\t20\tAR\tBR\t3\t0\tF\t\t11\t1950\t1889\t0\t−\t5\t7\n"
        "1914\t09\t27\tAR\tBR\t0\t1\tF\t\t-21\t1910\t1910\t0\t−\t5\t7\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as f:
        f.write(tsv_content)
        temp_path = f.name

    try:
        df = parse_elo_tsv(temp_path)
        assert len(df) == 2
        assert df.iloc[0]['year'] == '1914'
        assert df.iloc[0]['month'] == '09'
        assert df.iloc[0]['day'] == '20'
        assert df.iloc[0]['team_a_code'] == 'AR'
        assert df.iloc[0]['team_b_code'] == 'BR'
        assert df.iloc[0]['points_change'] == 11
        assert df.iloc[1]['points_change'] == -21
        assert df.iloc[0]['date'] == pd.Timestamp('1914-09-20')
    finally:
        os.unlink(temp_path)

def test_parse_elo_tsv_comma():
    # Mock a comma-separated file without headers
    csv_content = (
        "2020,01,15,AR,BR,2,1,F,,5,1955,1894,0,-,5,7\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        df = parse_elo_tsv(temp_path)
        assert len(df) == 1
        assert df.iloc[0]['year'] == '2020'
        assert df.iloc[0]['points_change'] == 5
        assert df.iloc[0]['date'] == pd.Timestamp('2020-01-15')
    finally:
        os.unlink(temp_path)

def test_parse_elo_tsv_with_unicode_minus():
    # Test U+2212 minus sign
    tsv_content = (
        "2002\t06\t30\tBR\tDE\t2\t0\tWC\t\t−15\t2010\t1980\t0\t0\t1\t2\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as f:
        f.write(tsv_content)
        temp_path = f.name

    try:
        df = parse_elo_tsv(temp_path)
        assert len(df) == 1
        assert df.iloc[0]['points_change'] == -15
    finally:
        os.unlink(temp_path)

def test_parse_elo_tsv_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_elo_tsv("non_existent_file.tsv")

def test_merge_elo_and_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create a mock canonical_teams.json
        mapping = {
            "Argentina": {"code": "AR", "canonical_name": "Argentina", "elo_file": "Argentina.tsv"},
            "Brazil": {"code": "BR", "canonical_name": "Brazil", "elo_file": "Brazil.tsv"}
        }
        mapping_path = os.path.join(tmpdir, "canonical_teams.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f)
            
        # 2. Create mock Elo files
        # Brazil.tsv
        br_content = (
            "2014\t07\t08\tBR\tDE\t1\t7\tWC\t\t−30\t1950\t2050\t0\t0\t3\t2\n"
            "2014\t10\t11\tBR\tAR\t2\t0\tF\tCN\t15\t1965\t2000\t0\t0\t3\t4\n"
        )
        # Argentina.tsv
        ar_content = (
            "2014\t07\t09\tNL\tAR\t0\t0\tWC\tBR\t10\t2020\t2015\t0\t0\t1\t4\n"
            "2014\t10\t11\tBR\tAR\t2\t0\tF\tCN\t15\t1965\t2000\t0\t0\t3\t4\n"
        )
        
        elo_dir = os.path.join(tmpdir, "elo")
        os.makedirs(elo_dir)
        with open(os.path.join(elo_dir, "Brazil.tsv"), 'w', encoding='utf-8') as f:
            f.write(br_content)
        with open(os.path.join(elo_dir, "Argentina.tsv"), 'w', encoding='utf-8') as f:
            f.write(ar_content)
            
        # 3. Create mock results.csv
        results_content = (
            "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
            "2014-10-11,Brazil,Argentina,2,0,Friendly,Beijing,China,TRUE\n"
            "2014-11-15,Brazil,Germany,1,1,Friendly,Salvador,Brazil,FALSE\n" # Germany not in canonical teams
        )
        results_path = os.path.join(tmpdir, "results.csv")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(results_content)
            
        # 4. Call merge_elo_and_results
        df = merge_elo_and_results(results_path, mapping_path, elo_dir)
        
        # 5. Assertions
        assert len(df) == 1 # Only Brazil vs Argentina should be kept
        row = df.iloc[0]
        assert row['home_team'] == 'Brazil'
        assert row['away_team'] == 'Argentina'
        assert row['home_code'] == 'BR'
        assert row['away_code'] == 'AR'
        
        # Brazil Elo before 2014-10-11:
        # The match occurred on 2014-10-11.
        # Brazil Elo After = 1965, points_change = 15.
        # Since Brazil is home team (BR) and Team A is BR in the row (BR vs AR):
        # home_elo_before = 1965 - 15 = 1950.
        # Argentina Elo After = 2000, points_change = 15.
        # Since Argentina is away team (AR) and Team B is AR in the row:
        # away_elo_before = 2000 + 15 = 2015.
        assert row['home_elo_before'] == 1950
        assert row['away_elo_before'] == 2015
