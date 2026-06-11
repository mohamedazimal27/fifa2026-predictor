import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import time
import os
import random

# Import project simulator and features
from src.simulator import TournamentSimulator, GROUPS_2026
from src.features import compute_decay_form

# Set page config
st.set_page_config(
    page_title="FIFA 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Font and general styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background colors */
    .stApp {
        background-color: #0d0f18;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121524;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header gradient */
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Tab active indicator */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00f2fe;
        border-color: rgba(0, 242, 254, 0.3);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 242, 254, 0.15) !important;
        color: #00f2fe !important;
        border-color: #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)

COACH_STATS = {
    "Lionel Scaloni": {"win_rate": 0.68, "intl_win_rate": 0.68, "world_cups": 1, "trophies": ["World Cup 2022", "Copa América 2021, 2024"]},
    "Didier Deschamps": {"win_rate": 0.64, "intl_win_rate": 0.64, "world_cups": 3, "trophies": ["World Cup 2018", "Nations League 2021"]},
    "Dorival Júnior": {"win_rate": 0.58, "intl_win_rate": 0.55, "world_cups": 0, "trophies": ["Copa Libertadores 2022"]},
    "Julian Nagelsmann": {"win_rate": 0.62, "intl_win_rate": 0.55, "world_cups": 0, "trophies": ["Bundesliga 2021-22"]},
    "Luis de la Fuente": {"win_rate": 0.72, "intl_win_rate": 0.75, "world_cups": 0, "trophies": ["Euro 2024", "Nations League 2023"]},
    "Thomas Tuchel": {"win_rate": 0.60, "intl_win_rate": 0.60, "world_cups": 0, "trophies": ["Champions League 2021"]},
    "Luciano Spalletti": {"win_rate": 0.57, "intl_win_rate": 0.55, "world_cups": 0, "trophies": ["Serie A 2022-23"]},
    "Domenico Tedesco": {"win_rate": 0.59, "intl_win_rate": 0.62, "world_cups": 0, "trophies": ["DFB-Pokal 2021-22"]},
    "Roberto Martínez": {"win_rate": 0.66, "intl_win_rate": 0.70, "world_cups": 2, "trophies": ["FA Cup 2013"]},
    "Ronald Koeman": {"win_rate": 0.56, "intl_win_rate": 0.58, "world_cups": 0, "trophies": ["Copa del Rey 2021"]},
    "Gareth Southgate": {"win_rate": 0.61, "intl_win_rate": 0.61, "world_cups": 2, "trophies": ["None"]}
}

def get_coach_display_stats(coach_name: str) -> dict:
    return COACH_STATS.get(coach_name, {
        "win_rate": 0.50,
        "intl_win_rate": 0.48,
        "world_cups": 0,
        "trophies": []
    })

# Helper function to get base simulator
@st.cache_resource
def get_simulator():
    return TournamentSimulator()

# Warm the simulator
try:
    sim = get_simulator()
except Exception as e:
    st.error(f"Failed to load tournament simulator: {e}")
    st.stop()

# Cache Monte Carlo results in session state
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None
if 'mc_num_sims' not in st.session_state:
    st.session_state.mc_num_sims = 0

# Sidebar controls
st.sidebar.markdown("<h3 style='color: #00f2fe; font-weight:700;'>Simulation Parameters</h3>", unsafe_allow_html=True)
num_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=10000, value=2000, step=100)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

run_button = st.sidebar.button("⚡ Run Tournament Simulation", use_container_width=True)

if run_button or st.session_state.mc_results is None:
    # Set seed
    np.random.seed(seed)
    random.seed(seed)
    
    with st.spinner(f"Simulating {num_simulations:,} tournaments..."):
        t0 = time.time()
        st.session_state.mc_results = sim.run_monte_carlo(num_simulations=num_simulations)
        st.session_state.mc_num_sims = num_simulations
        st.sidebar.success(f"Executed in {time.time() - t0:.2f} seconds!")

# Banner
st.markdown("<div class='header-title'>FIFA 2026 World Cup Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Sports Analytics & Machine Learning Tournament Simulation Dashboard</div>", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tournament Overview", 
    "⚔️ Match Predictor", 
    "🔍 Team Analysis", 
    "🎮 Scenario Simulator", 
    "⚙️ Model Transparency"
])

# Create list of all 48 teams
all_teams = sorted(list(sim.starting_elos.keys()))

# ----------------- TAB 1: TOURNAMENT OVERVIEW -----------------
with tab1:
    st.markdown("### 🏆 World Cup 2026 Forecast")
    
    if st.session_state.mc_results:
        probs = st.session_state.mc_results
        
        # Sort teams by champion probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1]['champion'], reverse=True)
        
        # Create columns
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("<div class='glass-card'><h4>Top 15 Championship Contenders</h4>", unsafe_allow_html=True)
            top_15_teams = [t for t, _ in sorted_probs[:15]]
            top_15_champs = [probs[t]['champion'] * 100 for t in top_15_teams]
            
            fig_champ = px.bar(
                x=top_15_champs,
                y=top_15_teams,
                orientation='h',
                labels={'x': 'Championship Probability (%)', 'y': 'Team'},
                color=top_15_champs,
                color_continuous_scale='tealgrn',
                template='plotly_dark'
            )
            fig_champ.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=450,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_champ, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'><h4>Progression Probabilities by Stage (Top 10)</h4>", unsafe_allow_html=True)
            top_10_teams = [t for t, _ in sorted_probs[:10]]
            stages = ['group_stage_exit', 'r32_exit', 'r16_exit', 'qf_exit', 'sf_exit', 'third_place', 'runner_up', 'champion']
            stage_labels = ['Group Stage Exit', 'R32 Exit', 'R16 Exit', 'QF Exit', 'SF Exit', 'Third Place', 'Runner-up', 'Champion']
            
            fig_prog = go.Figure()
            for stage, label in zip(stages, stage_labels):
                fig_prog.add_trace(go.Bar(
                    name=label,
                    x=top_10_teams,
                    y=[probs[team][stage] * 100 for team in top_10_teams]
                ))
            fig_prog.update_layout(
                barmode='stack',
                template='plotly_dark',
                height=450,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_prog, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Champion DNA Comparison
        st.markdown("<div class='glass-card'><h4>🧬 Champion DNA: Core Metrics Comparison</h4>", unsafe_allow_html=True)
        dna_teams = [t for t, _ in sorted_probs[:5]]
        dna_rows = []
        for team in dna_teams:
            tf = sim.team_features[team]
            coach_info = sim.lookup_system.lookup(team, "2026-06-11")
            dna_rows.append({
                "Team": team.replace("_", " "),
                "Starting Elo": int(tf['elo']),
                "Squad Quality Score": f"{tf['squad_quality']:.2f}",
                "Coach": coach_info['coach'],
                "Coach Tenure (Years)": f"{coach_info['tenure_days'] / 365.0:.1f}",
                "Host Advantage": "Yes" if tf['host_advantage'] == 1 else "No",
                "Championship Prob": f"{probs[team]['champion']:.2%}"
            })
        st.table(pd.DataFrame(dna_rows))
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: MATCH PREDICTOR -----------------
with tab2:
    st.markdown("### ⚔️ Pairwise Match Predictor")
    st.write("Simulate a direct head-to-head matchup between any two teams using the calibrated model.")
    
    col_t1, col_vs, col_t2 = st.columns([2, 1, 2])
    
    with col_t1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        team_a = st.selectbox("Select Team A", all_teams, index=all_teams.index("Brazil"))
        
        # Display team details
        tf_a = sim.team_features[team_a]
        st.metric("Elo Rating", int(tf_a['elo']))
        st.metric("Squad Quality", f"{tf_a['squad_quality']:.2f}")
        
        fatigue_a = st.slider("Team A Fatigue Level", min_value=0, max_value=5, value=0, key="fat_a")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 80px; color: #00f2fe;'>VS</h1>", unsafe_allow_html=True)
        
        # Match location settings
        location_type = st.radio(
            "Match Location",
            options=["Neutral Venue", "Team A Home/Host", "Team B Home/Host"],
            index=0
        )
        
    with col_t2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        team_b = st.selectbox("Select Team B", all_teams, index=all_teams.index("Argentina"))
        
        # Display team details
        tf_b = sim.team_features[team_b]
        st.metric("Elo Rating", int(tf_b['elo']))
        st.metric("Squad Quality", f"{tf_b['squad_quality']:.2f}")
        
        fatigue_b = st.slider("Team B Fatigue Level", min_value=0, max_value=5, value=0, key="fat_b")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if team_a == team_b:
        st.warning("Please select two different teams to run a matchup simulation.")
    else:
        # Construct row
        elo_a = tf_a['elo'] - fatigue_a * 20.0
        elo_b = tf_b['elo'] - fatigue_b * 20.0
        
        # Handle custom location advantage
        host_a = 1 if (location_type == "Team A Home/Host") else 0
        host_b = 1 if (location_type == "Team B Home/Host") else 0
        
        host_advantage_home = 1 if host_a == 1 and host_b == 0 else 0
        host_advantage_away = 1 if host_b == 1 and host_a == 0 else 0
        
        row = {
            'elo_diff': elo_a - elo_b,
            'home_elo': elo_a,
            'away_elo': elo_b,
            'home_form': tf_a['form'],
            'away_form': tf_b['form'],
            'form_diff': tf_a['form'] - tf_b['form'],
            'home_squad_quality': tf_a['squad_quality'],
            'away_squad_quality': tf_b['squad_quality'],
            'squad_quality_diff': tf_a['squad_quality'] - tf_b['squad_quality'],
            'home_interim_coach': tf_a['is_interim'],
            'away_interim_coach': tf_b['is_interim'],
            'home_advantage': host_a,
            'host_advantage_home': host_advantage_home,
            'host_advantage_away': host_advantage_away
        }
        
        df_match = pd.DataFrame([row])[sim.feature_cols]
        base_preds = sim.base_model.predict_proba(df_match)
        cal_preds = sim.calibrator.predict_proba(base_preds)[0]
        
        # Display Outcome Probabilities
        st.markdown("<div class='glass-card'><h3>🔮 Calibrated Outcome Probabilities</h3>", unsafe_allow_html=True)
        p_home, p_draw, p_away = cal_preds[0], cal_preds[1], cal_preds[2]
        
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric(f"{team_a.replace('_', ' ')} Win", f"{p_home:.1%}")
        col_p2.metric("Draw", f"{p_draw:.1%}")
        col_p3.metric(f"{team_b.replace('_', ' ')} Win", f"{p_away:.1%}")
        
        # Scoreline simulation
        sim_scores = []
        for _ in range(5000):
            sim_scores.append(sim.simulate_match_goals(p_home, p_draw, p_away))
        
        score_counts = {}
        for s1, s2, _ in sim_scores:
            score = (s1, s2)
            score_counts[score] = score_counts.get(score, 0) + 1
            
        top_score = max(score_counts.items(), key=lambda x: x[1])[0]
        top_score_pct = score_counts[top_score] / 5000
        
        st.markdown(f"<h4>Most Likely Scoreline: <span style='color:#00f2fe;'>{team_a.replace('_', ' ')} {top_score[0]} - {top_score[1]} {team_b.replace('_', ' ')}</span> ({top_score_pct:.1%} probability)</h4>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: TEAM ANALYSIS -----------------
with tab3:
    st.markdown("### 🔍 Team Strength & Coach Profile")
    selected_team = st.selectbox("Select a Team for Analysis", all_teams)
    
    tf_team = sim.team_features[selected_team]
    coach_info = sim.lookup_system.lookup(selected_team, "2026-06-11")
    
    col_ta1, col_ta2 = st.columns(2)
    
    with col_ta1:
        st.markdown("<div class='glass-card'><h4>📈 Strength Profile</h4>", unsafe_allow_html=True)
        st.write(f"**Starting Elo Rating:** {int(tf_team['elo'])}")
        st.write(f"**Squad Quality Score:** {tf_team['squad_quality']:.2f}")
        st.write(f"**Host Status:** {'Host Country 🇺🇸🇲🇽🇨🇦' if tf_team['host_advantage'] == 1 else 'Normal Competitor'}")
        
        # Find group details
        team_group = "Unknown"
        for g, teams in GROUPS_2026.items():
            if selected_team in teams:
                team_group = g
                break
        st.write(f"**Group:** Group {team_group}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_ta2:
        st.markdown("<div class='glass-card'><h4>👔 Coach Profile</h4>", unsafe_allow_html=True)
        c_stats = get_coach_display_stats(coach_info['coach'])
        st.write(f"**Coach Name:** {coach_info['coach']}")
        st.write(f"**Tenure:** {coach_info['tenure_days'] / 365.0:.2f} years")
        st.write(f"**Win Rate:** {c_stats['win_rate']:.1%}")
        st.write(f"**International Win Rate:** {c_stats['intl_win_rate']:.1%}")
        st.write(f"**World Cup Experience:** {c_stats['world_cups']} prior tournaments")
        st.write(f"**Major Trophies:** {', '.join(c_stats['trophies']) if c_stats['trophies'] else 'None'}")
        st.write(f"**Is Interim Coach:** {'Yes ⚠️' if coach_info['is_interim'] else 'No'}")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 4: SCENARIO SIMULATOR -----------------
with tab4:
    st.markdown("### 🎮 Custom Scenario Simulator")
    st.write("Edit a team's core attributes in memory and run a quick simulation to see how it alters their tournament path.")
    
    scen_team = st.selectbox("Select Team to Modify", all_teams, key="scen_t")
    
    col_sc1, col_sc2 = st.columns(2)
    
    tf_sc = sim.team_features[scen_team]
    coach_sc = sim.lookup_system.lookup(scen_team, "2026-06-11")
    
    with col_sc1:
        st.markdown("<div class='glass-card'><h4>⚙️ Team Attributes</h4>", unsafe_allow_html=True)
        edited_elo = st.slider("Starting Elo Rating", min_value=1000, max_value=2200, value=int(tf_sc['elo']))
        edited_squad = st.slider("Squad Quality Score", min_value=0.0, max_value=1.0, value=float(tf_sc['squad_quality']))
        edited_host = st.checkbox("Has Host Advantage", value=(tf_sc['host_advantage'] == 1))
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_sc2:
        st.markdown("<div class='glass-card'><h4>👔 Coach Attributes</h4>", unsafe_allow_html=True)
        c_sc_stats = get_coach_display_stats(coach_sc['coach'])
        edited_coach_win = st.slider("Coach Win Rate", min_value=0.0, max_value=1.0, value=float(c_sc_stats['win_rate']))
        edited_tenure = st.slider("Coach Tenure (Years)", min_value=0.0, max_value=15.0, value=float(coach_sc['tenure_days'] / 365.0))
        edited_interim = st.checkbox("Is Interim Coach", value=bool(coach_sc['is_interim']))
        st.markdown("</div>", unsafe_allow_html=True)
        
    run_scen = st.button("🚀 Run Scenario Simulation (1,000 Runs)", use_container_width=True)
    
    if run_scen:
        # Save originals
        orig_elo = sim.starting_elos[scen_team]
        orig_tf = sim.team_features[scen_team].copy()
        
        # Save original matchup cache
        orig_cache = {}
        for key, val in sim.matchup_cache.items():
            t1, t2, f1, f2 = key
            if t1 == scen_team or t2 == scen_team:
                orig_cache[key] = val
                
        # Apply edits
        sim.starting_elos[scen_team] = float(edited_elo)
        sim.team_features[scen_team]['elo'] = float(edited_elo)
        sim.team_features[scen_team]['squad_quality'] = float(edited_squad)
        sim.team_features[scen_team]['is_interim'] = 1 if edited_interim else 0
        sim.team_features[scen_team]['host_advantage'] = 1 if edited_host else 0
        
        # Warm cache for edited team
        for group, teams in GROUPS_2026.items():
            for other_team in teams:
                if other_team == scen_team:
                    continue
                for f1 in range(6):
                    for f2 in range(6):
                        sim.matchup_cache[(scen_team, other_team, f1, f2)] = sim.predict_match(scen_team, other_team, f1, f2)
                        sim.matchup_cache[(other_team, scen_team, f1, f2)] = sim.predict_match(other_team, scen_team, f1, f2)
                        
        # Run simulation
        with st.spinner("Simulating scenario..."):
            scen_results = sim.run_monte_carlo(num_simulations=1000)
            
        # Restore originals
        sim.starting_elos[scen_team] = orig_elo
        sim.team_features[scen_team] = orig_tf
        for key, val in orig_cache.items():
            sim.matchup_cache[key] = val
            
        # Display comparison
        st.markdown("<div class='glass-card'><h3>📊 Scenario Results Comparison</h3>", unsafe_allow_html=True)
        
        orig_probs = st.session_state.mc_results[scen_team] if st.session_state.mc_results else None
        new_probs = scen_results[scen_team]
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("<h4>Before Scenario</h4>", unsafe_allow_html=True)
            if orig_probs:
                st.write(f"🏆 **Champion Probability:** {orig_probs['champion']:.2%}")
                st.write(f"🥈 **Runner-up Probability:** {orig_probs['runner_up']:.2%}")
                st.write(f"🥉 **Third Place Probability:** {orig_probs['third_place']:.2%}")
                st.write(f"🚪 **Group Stage Exit:** {orig_probs['group_stage_exit']:.2%}")
            else:
                st.write("Run the main tournament simulation to compare.")
                
        with col_res2:
            st.markdown("<h4>After Scenario</h4>", unsafe_allow_html=True)
            st.write(f"🏆 **Champion Probability:** {new_probs['champion']:.2%}")
            st.write(f"🥈 **Runner-up Probability:** {new_probs['runner_up']:.2%}")
            st.write(f"🥉 **Third Place Probability:** {new_probs['third_place']:.2%}")
            st.write(f"🚪 **Group Stage Exit:** {new_probs['group_stage_exit']:.2%}")
            
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 5: MODEL TRANSPARENCY -----------------
with tab5:
    st.markdown("### ⚙️ Model Transparency & Validation Metrics")
    
    # Load and show benchmark metrics
    if os.path.exists("models/benchmark_metrics.json"):
        with open("models/benchmark_metrics.json", "r") as f:
            metrics = json.load(f)
            
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("<div class='glass-card'><h4>📉 Log Loss Performance</h4>", unsafe_allow_html=True)
            st.write("**Training Set (2000-2015):**")
            st.write(f"  Uncalibrated: {metrics['uncalibrated']['train_loss']:.4f} | Calibrated: {metrics['calibrated']['train_loss']:.4f}")
            st.write("**Val B Set (2016-2017):**")
            st.write(f"  Uncalibrated: {metrics['uncalibrated']['val_b_loss']:.4f} | Calibrated: {metrics['calibrated']['val_b_loss']:.4f}")
            st.write("**Test Set (2018):**")
            st.write(f"  Uncalibrated: {metrics['uncalibrated']['test_loss']:.4f} | Calibrated: {metrics['calibrated']['test_loss']:.4f}")
            st.write("**Holdout Set (2022):**")
            st.write(f"  Uncalibrated: {metrics['uncalibrated']['holdout_loss']:.4f} | Calibrated: {metrics['calibrated']['holdout_loss']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_m2:
            st.markdown("<div class='glass-card'><h4>🎯 Classification Accuracy</h4>", unsafe_allow_html=True)
            st.write(f"**Test Set (2018) Accuracy:** {metrics['calibrated']['test_accuracy']:.2%}")
            st.write(f"**Holdout Set (2022) Accuracy:** {metrics['calibrated']['holdout_accuracy']:.2%}")
            st.write("")
            st.write("*Accuracy is computed by taking the argmax of outcome probabilities.*")
            st.markdown("</div>", unsafe_allow_html=True)
            
    # Feature Importance Chart
    st.markdown("<div class='glass-card'><h4>📊 Feature Importance (XGBoost)</h4>", unsafe_allow_html=True)
    importances = sim.base_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': sim.feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(
        feat_imp_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='plasma',
        template='plotly_dark'
    )
    fig_imp.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
