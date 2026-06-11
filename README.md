# 🏆 FIFA 2026 World Cup Predictor

A machine learning and Monte Carlo simulation engine designed to model and simulate the expanded **48-team FIFA 2026 World Cup** in North America. Built using **XGBoost** for match outcome probability estimation and **Platt Scaling** for probability calibration, this application provides a statistical framework for tournament modeling and includes an interactive **Streamlit** dashboard.

---


## 📊 Model Performance Metrics

The model achieves high accuracy and optimal probability calibration, evaluated across strict temporal splits:

| Metric | Uncalibrated XGBoost | Calibrated Model (Platt Scaling) |
| :--- | :---: | :---: |
| **Train Log Loss (2000–2013)** | 0.9091 | **0.8723** |
| **Validation B Log Loss (2016–2017)** | 0.9117 | **0.8769** |
| **Test Log Loss (2018)** | 0.9519 | **0.9179** |
| **Holdout Log Loss (2022)** | 0.9482 | **0.9015** |
| **Test Accuracy (2018)** | — | **56.09%** |
| **Holdout Accuracy (2022)** | — | **58.95%** |

---

## 🛠️ Feature Engineering Architecture

To capture the multi-dimensional nature of international football, the predictor leverages a **six-tier feature hierarchy** spanning relative strength, recent form, coach stats, squad depth, tournament context, and historic success.

### 🔹 Tier 1: Relative Strength (Foundation)
*   **Elo Rating Difference (`elo_diff`)**: $Elo_{Home} - Elo_{Away}$. The primary predictor of match outcomes.
*   **Fatigue Elo Penalty**: A heuristic simulation parameter representing how tournament fatigue might degrade performance. For every fatigue unit a team has accumulated in previous tournament stages, their effective Elo is penalized by **20 points** ($Elo_{effective} = Elo_{base} - 20 \times Fatigue$). *Note: This 20-point penalty is an arbitrary heuristic used to model schedule-induced fatigue and has not been statistically calibrated or validated against physical tracking data.*
*   **Squad Quality Difference (`squad_quality_diff`)**: Log-transformed squad market value difference modeled using a curated temporal index relative to continental peers.

### 🔹 Tier 2: Recent Form (Time-Decayed)
*   **Time-Decayed Outcomes (`form_win_rate_decay`)**: Form features are computed using an exponential decay function with a half-life of $\lambda = 365\text{ days}$. Recent matches carry significantly higher weight:
    $$W_i = \exp(-\lambda \cdot t_i)$$
    Where $t_i$ is the time elapsed since match $i$ in years.
*   **Form Difference (`form_diff`)**: $Form_{Home} - Form_{Away}$.

### 🔹 Tier 3: Coach Experience & Stability
*   **Interim/New Coach Flags (`home_interim_coach`, `away_interim_coach`)**: Flags if a coach's tenure at match date is under **180 days** (associated with tactical instability).
*   **Coach Success Ratings**: Mapped from historical coach performance registries to capture manager caliber. *Note: Due to data sparsity, teams outside the top 10 receive fallback confederation-level baseline values. This feature should be treated as a heuristic baseline rather than a fully validated predictor.*

### 🔹 Tier 4: Team Chemistry & Depth
*   **Squad Quality Ratings**: A temporal squad score rating (e.g. Argentina 2022: `1.28`, Argentina 2023–2026: `1.30`) mapped over distinct temporal bounds, reflecting the generation's depth.

### 🔹 Tier 5: Tournament Context
*   **Home Advantage (`home_advantage`)**: 1 if a country is playing on its home soil in a non-neutral match.
*   **Host Advantage (`host_advantage_home` & `host_advantage_away`)**: Specially configured for the 2026 tournament. USA, Mexico, and Canada receive localized host boosts when playing in North America.

### 🔹 Tier 6: Champion DNA
*   **Historic Indicators**: Past World Cup quarter-final/semi-final appearances, continental titles, and squad average age are utilized for visualization and profiling.

---

## 🔬 Statistical Validation & Splits

To avoid leakage and double-dipping, a **five-stage sequential temporal split** was used to build the pipeline:

1.  **Train Set (2000–2013)**: Trains the base XGBoost classifier on historic matches.
2.  **Validation Set A (2014–2015)**: Used to optimize hyperparameters (including exponential form decay half-lives).
3.  **Validation Set B (2016–2017)**: Used to fit the **Platt Scaling** model (Logistic Regression) to calibrate XGBoost output probabilities.
4.  **Test Set 1 (2018)**: Benchmarking set for evaluating log loss, accuracy, and calibration curves.
5.  **Test Set 2 / Holdout (2022)**: Complete out-of-sample holdout validation representing the 2022 Qatar World Cup.

---

## 🔮 Tournament Simulation Engine

The simulation engine models the expanded **48-team FIFA 2026 World Cup** following the official format:

### 1️⃣ Group Stage
*   Match outcomes are simulated using calibrated win-draw-loss probabilities.
*   Standings are determined deterministically using official FIFA tie-breakers:
    1.  Points
    2.  Goal Difference (GD)
    3.  Goals Scored (GS)
    4.  Head-to-Head (H2H) Points
    5.  H2H Goal Difference
    6.  H2H Goals Scored
    7.  Elo Rating (as the ultimate fallback)

### 2️⃣ Best Third-Place Routing
*   The **8 best third-placed teams** advance to the Round of 32.
*   Routing uses a programmatic, bipartite matching-inspired algorithm (`assign_third_places`) to assign third-place teams to their round of 32 opponents, avoiding group-stage rematches according to official FIFA guidelines.

### 3️⃣ Knockout Stage
*   Single-elimination brackets from the Round of 32 down to the Final and Third Place Playoff.
*   If a knockout match ends in a draw:
    *   **Extra Time** is simulated: Teams receive a **+1 fatigue unit** increase.
    *   **Penalty Shootout**: Modeled using a customized probability model taking into account the squad quality difference, Elo difference, and cumulative fatigue of both squads.

### 🏥 Injury Shock System
The simulation and predictor support an interactive **Injury Shock System** to simulate key player injuries. Applying a shock to a team penalizes their rating dynamically (Key: -30 Elo, World Class: -50 Elo, Indispensable: -80 Elo), updates squad quality, and instantly updates the matchup cache to re-run simulations.

---

## 🖥️ Streamlit Interactive Dashboard

The dashboard consists of six interactive tabs:

1.  **🏆 Tournament Overview**: Displays Monte Carlo simulation results after 1,000 to 10,000 iterations with 95% Confidence Intervals. Features Plotly charts for championship contenders, bracket progression probabilities, and Champion DNA overlays.
2.  **⚔️ Match Predictor**: A head-to-head calculator that outputs calibrated win/draw/loss probabilities with 95% Confidence Intervals for any two teams, allowing manual configuration of venue, host advantages, and fatigue.
3.  **🔍 Team Analysis**: Explores detailed team profiles, including active coach details, squad quality indexes, and tournament metrics.
4.  **🎮 Scenario Simulator**: A sandbox tab allowing users to modify a team's Elo, squad quality, or coach status in memory to see how it affects simulation outcomes with Before vs. After confidence interval comparisons.
5.  **⏪ Historical Replay**: Replays the 2018 and 2022 World Cups, comparing the model's calibrated predictions side-by-side with actual results and displaying overall log loss and accuracy metrics.
6.  **⚙️ Model Transparency**: Displays Platt Scaling calibration curves (Reliability Diagrams), validation log loss metrics, feature importance, and KS-test feature drift analyses.

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

Clone the repository and prepare a virtual environment:

```bash
# Clone the repository
git clone https://github.com/mohamedazimal27/fifa2026-predictor.git
cd fifa2026-predictor

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Local Model Artifacts

Model artifacts are generated locally and intentionally excluded from the public repository. Train the model before launching the dashboard:

```bash
PYTHONPATH="." venv/bin/python -m src.models.train
PYTHONPATH="." venv/bin/python -m src.models.evaluate_backtests
```

This creates the local `models/` directory used by the Streamlit app.

### 3. Running the Streamlit App

Run the Streamlit server. It is recommended to clear `PYTHONPATH` to prevent conflicts with other system packages:

```bash
PYTHONPATH="" venv/bin/streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

### 4. Running Unit Tests

The codebase includes full test coverage for the features, models, simulators, and routing logic:

```bash
PYTHONPATH="." venv/bin/pytest
```

---

## 📁 Repository Structure

```
├── app.py                     # Streamlit dashboard entry point
├── requirements.txt           # Project dependencies
├── src/
│   ├── features.py            # Feature engineering and decay computations
│   ├── simulator.py           # World Cup tournament and shootout simulator
│   ├── third_place_router.py  # Best third-place assignment logic
│   ├── data_pipeline/
│   │   ├── data_loader.py     # Loader for matches and daily Elos
│   │   ├── curated_lookup.py  # Temporal team metadata lookup
│   │   └── download_elo.py    # Elo data download helper
│   ├── models/
│   │   ├── train.py           # Model training, calibration, and metrics export
│   │   └── evaluate_backtests.py
│   └── tuning/
│       └── decay_search.py    # Form-decay tuning utility
├── data/
│   ├── results.csv            # Historic match results (2000-2024)
│   ├── shootouts.csv          # Penalty shootout logs
│   ├── elo/                   # Daily Elo ratings for 150+ nations
│   ├── curated_teams.json     # Temporal squad quality & coach timelines
│   └── canonical_teams.json   # Team name/code/file mapping
├── models/                    # Local generated artifacts, ignored by Git
└── tests/                     # pytest suite
```
