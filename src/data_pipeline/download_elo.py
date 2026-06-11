import os
import json
import urllib.request
import urllib.error
import time
import pandas as pd

def download_and_map_elo():
    # Paths
    results_path = 'data/results.csv'
    en_teams_path = 'data/en.teams.tsv'
    elo_dir = 'data/elo'
    canonical_teams_path = 'data/canonical_teams.json'
    
    os.makedirs(elo_dir, exist_ok=True)
    
    # 1. Get unique teams since 2000 from results.csv
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    df = pd.read_csv(results_path)
    df['date'] = pd.to_datetime(df['date'])
    df_2000 = df[df['date'].dt.year >= 2000]
    unique_teams = sorted(list(set(df_2000['home_team']).union(set(df_2000['away_team']))))
    print(f"Found {len(unique_teams)} unique teams since 2000 in results.csv")
    
    # 2. Download en.teams.tsv if not exists
    if not os.path.exists(en_teams_path):
        print("Downloading en.teams.tsv...")
        urllib.request.urlretrieve('https://www.eloratings.net/en.teams.tsv', en_teams_path)
        
    # 3. Parse en.teams.tsv to build name-to-code mapping
    teams_list = []
    with open(en_teams_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                code = parts[0]
                name = parts[1]
                aliases = parts[2:]
                teams_list.append({'code': code, 'name': name, 'aliases': aliases})
                
    # Build a helper mapping dictionary
    name_to_info = {}
    for t in teams_list:
        info = {'code': t['code'], 'canonical_name': t['name']}
        name_to_info[t['name'].lower()] = info
        for alias in t['aliases']:
            name_to_info[alias.lower()] = info
            name_to_info[alias.lower().replace('&', 'and')] = info
            name_to_info[alias.lower().replace('and', '&')] = info
        name_to_info[t['name'].lower().replace('&', 'and')] = info
        name_to_info[t['name'].lower().replace('and', '&')] = info
        
    # Manual overrides/custom mappings for known mismatches
    custom_mappings = {
        "american samoa": {"code": "AS", "canonical_name": "Eastern Samoa"},
        "czech republic": {"code": "CZ", "canonical_name": "Czechia"},
        "macau": {"code": "MO", "canonical_name": "Macao"},
        "republic of ireland": {"code": "IE", "canonical_name": "Ireland"},
        "são tomé and príncipe": {"code": "ST", "canonical_name": "Sao Tome and Principe"},
        "timor-leste": {"code": "TL", "canonical_name": "East Timor"},
        "united states virgin islands": {"code": "VI", "canonical_name": "US Virgin Islands"},
    }
    
    for k, v in custom_mappings.items():
        name_to_info[k] = v
        
    # 4. Map teams and download TSVs
    canonical_mapping = {}
    missing_teams = []
    
    for team in unique_teams:
        t_lower = team.lower()
        info = None
        if t_lower in name_to_info:
            info = name_to_info[t_lower]
        else:
            # Try some string matching or replacements
            # If the team name has a slash or hyphen, try splitting
            for sep in ['/', '-']:
                if sep in t_lower:
                    parts = t_lower.split(sep)
                    for part in parts:
                        part_strip = part.strip()
                        if part_strip in name_to_info:
                            info = name_to_info[part_strip]
                            break
                if info:
                    break
        
        if info:
            # Determine filename
            import unicodedata
            canonical_name = info['canonical_name']
            clean_name = ''.join(c for c in unicodedata.normalize('NFD', canonical_name) if unicodedata.category(c) != 'Mn')
            filename = clean_name.replace(' ', '_')
            tsv_filename = f"{filename}.tsv"
            
            canonical_mapping[team] = {
                "code": info['code'],
                "canonical_name": canonical_name,
                "elo_file": tsv_filename
            }
        else:
            missing_teams.append(team)
            
    print(f"Successfully mapped {len(canonical_mapping)} teams.")
    print(f"Could not map {len(missing_teams)} teams (likely non-FIFA/regional teams).")
    
    # Save canonical mapping
    with open(canonical_teams_path, 'w', encoding='utf-8') as f:
        json.dump(canonical_mapping, f, indent=4, ensure_ascii=False)
    print(f"Saved canonical team mapping to {canonical_teams_path}")
    
    # 5. Download TSV files
    downloaded_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Sort by team name to keep it neat
    for team, details in sorted(canonical_mapping.items()):
        tsv_name = details['elo_file']
        dest_path = os.path.join(elo_dir, tsv_name)
        
        if os.path.exists(dest_path):
            skipped_count += 1
            continue
            
        # URL encode the filename part of the URL
        url_encoded_name = urllib.parse.quote(tsv_name)
        url = f"https://www.eloratings.net/{url_encoded_name}"
        
        print(f"Downloading Elo TSV for {team} ({tsv_name}) from {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read()
            with open(dest_path, 'wb') as f:
                f.write(content)
            downloaded_count += 1
            # Rate limiting friendly: sleep briefly between downloads
            time.sleep(0.2)
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code} downloading {team}: {e.reason}")
            failed_count += 1
        except Exception as e:
            print(f"Error downloading {team}: {e}")
            failed_count += 1
            
    print("\nDownload Summary:")
    print(f"Downloaded: {downloaded_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Failed: {failed_count}")
    
if __name__ == "__main__":
    download_and_map_elo()
