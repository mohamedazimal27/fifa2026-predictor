FIFA 2026 World Cup Predictor - Final Implementation Plan
This is the definitive, execution-ready implementation plan for the FIFA 2026 World Cup Predictor. It has been reviewed through the Google Colab and Streamlit Cloud lenses, scoring an 8.2 / 10 on readiness.

1. Executive Summary & Readiness Score
Readiness Score: 8.2 / 10

The Core Architecture is Sound: Dual prediction paths (live match-day for historical, pre-computed in-memory lookup for 2026), a strict 5-way split to avoid statistical double-dipping, and self-contained repository Elo caching.
Go/No-Go Conditions: We proceed with execution under strict constraints (e.g., fallback to 5K runs if 10K is slow; fallback to 10 curated teams if curation exceeds Day 3; fallback to temperature scaling if Platt is unstable).
2. Dimension-by-Dimension Readiness Assessment
Dimension	Rating	Key Execution Details
Statistical Rigor	9 / 10	No double-dipping. Proper splits: Train (2000–2013), Val A (2014–2015), Val B (2016–2017), Test (2018), Holdout (2022). We will explicitly verify Platt calibration quality on Test Set 1 (2018) before final benchmark declarations.
Data Architecture	9 / 10	Self-contained workspace in data/ using cached TSV files for all teams appearing in matches during the training/evaluation window ($\approx 150$ teams).
Technical Feasibility	8 / 10	Pre-computed matchup probabilities for 2,256 matchups make simulation speed targets achievable. Dual execution paths separate historical backtests from the 2026 simulation.
Coach Data Realism	7 / 10	Start with top 10 curated teams (Brazil, Argentina, France, Germany, Spain, England, Italy, Netherlands, Portugal, Belgium) on Day 3. Fall back to confederation-level baselines for the rest, expanding incrementally.
Third-Place Router	7 / 10	Programmatic derivation of the bracket matching combinations based on FIFA tournament rules. If programmatic routing is not complete by Day 10, a simplified deterministic router will be deployed.
UI & Deployment	8 / 10	Streamlit Cloud deployment with pre-trained model and pre-computed SHAP values committed to the repo. Static markdown-based bracket visualization is used to ensure UI stability.
Risk Mitigation	9 / 10	Explicit risk table with phased timelines and fail-safes. Phased Day 1–3 execution prevents pipeline regression.
Timeline Honesty	8 / 10	Realistic 14-day sprint. Curation and routing are prioritized as key time-sinks.
Testing Strategy	7 / 10	Property-based testing for group standings/routing. Calibration binning verification. Adds testing for is_interim behavior and host advantage edge cases (e.g. Mexico playing in Toronto).
Documentation/Portfolio	8 / 10	A clean, sectioned colab_exploration.ipynb with checkpoint guards. An explicit limitations section will be added to the main project README.
3. Phased Execution Timeline (Days 1–3 Focus)
To establish a solid foundation before building downstream features:

Day 1: Data Verification & Inspection
Download Mart Jürisoo's match datasets (results.csv, shootouts.csv).
Download 10 sample country TSVs from eloratings.net.
Build parse_elo_tsv() with automatic format detection (headers, tabs vs. commas, date format normalization) rather than hardcoded assumptions.
Day 2: Data Merging & Validation
Build canonical_teams.json for name resolution.
Merge Jürisoo matches + Elo ratings for the 2018 World Cup.
Manually cross-check Elo ratings for 5 random matches against the official eloratings.net website to verify date boundaries and ensure no future leakage.
Day 3: Top-10 Coach Curation
Populate curated_teams.json for the top 10 historical teams.
For the remaining $\sim 50$ teams, set up automated confederation-level fallback baselines (average squad values and ages, baseline coach tenure, win rates).
4. Project Splits & Data Strategy
Match Timeline: 2000 ───────────────────────────────────────────────> 2026
[Train Set: 2000-2013] [Val A: 2014-2015] [Val B: 2016-2017] [Test: 2018] [Holdout: 2022]
                       ├── Decay Search   ├── Platt Scaling
Validation A (2014–2015): Used to tune the exponential decay half-life parameter ($\lambda \in [30, 60, 90, 120, 180, 365]$ days).
Validation B (2016–2017): Used to fit the Platt Scaling calibration model (logistic regression).
Fail-safe: If Validation B does not contain enough draw samples to fit a stable sigmoid, we will fall back to Temperature Scaling.
Test Set 1 (2018): 64 matches used to benchmark Log Loss, Brier Score, and calibration curves.
Test Set 2 (2022): 64 matches kept as a final hold-out set, untouched until final evaluation.
5. Temporal Coach & Squad Database Schema
The curated data resides in data/curated_teams.json and uses a historical timeline model to reflect changing realities:

json
{
  "Brazil": {
    "historical_squads": [
      { "year": 2014, "squad_value_eur": 800000000, "average_age": 27.2 },
      { "year": 2018, "squad_value_eur": 950000000, "average_age": 28.1 },
      { "year": 2022, "squad_value_eur": 1050000000, "average_age": 27.9 },
      { "year": 2026, "squad_value_eur": 1100000000, "average_age": 26.5 }
    ],
    "coaches": [
      {
        "name": "Luiz Felipe Scolari",
        "start_date": "2012-11-28",
        "end_date": "2014-07-14",
        "career_win_rate": 0.55,
        "international_win_rate": 0.62,
        "prior_world_cups": 2,
        "major_trophies": ["World Cup 2002", "Copa América 1997"]
      },
      {
        "name": "Tite",
        "start_date": "2016-06-20",
        "end_date": "2022-12-09",
        "career_win_rate": 0.63,
        "international_win_rate": 0.73,
        "prior_world_cups": 1,
        "major_trophies": ["Copa América 2019"]
      },
      {
        "name": "Dorival Júnior",
        "start_date": "2024-01-10",
        "end_date": null,
        "career_win_rate": 0.58,
        "international_win_rate": 0.60,
        "prior_world_cups": 0,
        "major_trophies": ["Copa Libertadores 2022"]
      }
    ]
  }
}
Mitigation for Ambiguous Dates: If a coach's end date is ambiguous, we use the tournament end date as a proxy if it is within 30 days of departure.

6. What's Still Fragile & How We Fix It on the Fly
Fragile Element	Probability	Consequence	Mitigation
Elo TSV parsing	60%	Script crashes due to header/delimiter mismatches.	Parser uses try/except format detection. If parsing takes $> 4$ hours, we commit parsed raw data directly and proceed.
Third-place router	50%	Bracket combinations are mathematically complex to derive.	If not derived by Day 10, we deploy a simplified router that avoids same-group rematches and document the simplification.
Platt calibration quality	30%	Sigmoid fit is unstable due to draw sample counts.	Fall back to temperature scaling if Platt regression fails validation checks.
Streamlit Cloud memory	25%	Pre-computed lookup + SHAP cache exceeds the 1GB RAM limit.	Limit SHAP database caching to the top 20 teams and calculate other teams on-demand.
Simulation Speed	20%	10K runs exceed 30 seconds.	Reduce Monte Carlo simulation volume to 5,000 runs to maintain performance.
7. Go/No-Go Execution Gates
Gate 1: Start Day 1 with data verification and format-robust parsing. Do not write feature code yet.
Gate 2: If Elo parsing stalls on format nuances for more than 4 hours, bypass and commit a cleaned version manually.
Gate 3: If top-10 coach curation is not complete by Day 3, freeze the list at current progress and use confederation baselines for all remaining teams.
Gate 4: If the third-place bracket router is not fully verified by Day 10, implement a simplified version.
Gate 5: Adjust simulation volume (10K vs 5K) dynamically to guarantee runtime under 30 seconds.
8. Verification & Test Architecture
Pytest Suite (tests/):
Elo Lookup Validation: Verifies matching daily Elo scores against 5 validated historical data points.
Third-Place Routing Coverage: Property-based tests verifying all 495 combinations resolve to valid Round of 32 configurations.
Interim Coach Flag: Tests that tenure is set to 0 and baseline win rates apply when tenure is $< 180$ days.
Host Advantage Edge Cases: Verifies that Mexico playing in Canada gets host_advantage = 0, while Canada gets host_advantage = 1.
Calibration Binning Test: Evaluates calibration quality by binning match probabilities and asserting calibration error metrics.
9. Reproducibility: Colab Notebook Structure
The file colab_exploration.ipynb will be organized into 7 guarded, linear sections:

colab_exploration.ipynb
├── Section 1: Data Loading (caches datasets, parses Elo TSVs)
├── Section 2: Feature Engineering (computes decay form, merges metadata)
├── Section 3: Model Training (fits XGBoost + Platt Calibration, exports pickle)
├── Section 4: Backtesting 2018 (evaluates Log Loss and Brier Score on 2018 WC)
├── Section 5: Backtesting 2022 (evaluates Log Loss and Brier Score on 2022 WC)
├── Section 6: 2026 Simulation (runs Monte Carlo group/knockout simulator)
└── Section 7: Visualizations (displays Plotly inline performance charts)
Each section features a "skip if already saved" checkpoint guard to protect against Colab runtime disconnects.