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
    
    /* Interactive Walkthrough styling */
    .walkthrough-match-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .walkthrough-match-card:hover {
        border-color: rgba(0, 242, 254, 0.3);
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-2px);
    }
    .walkthrough-winner-card {
        border: 1px solid rgba(0, 242, 254, 0.3);
        background: rgba(0, 242, 254, 0.03);
    }
    .walkthrough-winner-card:hover {
        border-color: rgba(0, 242, 254, 0.6);
        background: rgba(0, 242, 254, 0.05);
    }
    .walkthrough-team-name {
        font-weight: 600;
        font-size: 1.05rem;
    }
    .walkthrough-team-winner {
        color: #00f2fe !important;
    }
    .walkthrough-score {
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
    }
    .walkthrough-prob-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 8px;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .walkthrough-bar-container {
        height: 6px;
        border-radius: 3px;
        display: flex;
        overflow: hidden;
        margin-top: 8px;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .walkthrough-bar-home {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
    }
    .walkthrough-bar-draw {
        background-color: rgba(255, 255, 255, 0.2);
    }
    .walkthrough-bar-away {
        background: linear-gradient(90deg, #f87171, #ef4444);
    }
    .walkthrough-badge {
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .walkthrough-badge-et {
        background-color: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .walkthrough-badge-pk {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

TEAM_FLAGS = {
    'Mexico': '🇲🇽',
    'South_Africa': '🇿🇦',
    'South_Korea': '🇰🇷',
    'Czechia': '🇨🇿',
    'Canada': '🇨🇦',
    'Bosnia_and_Herzegovina': '🇧🇦',
    'Qatar': '🇶🇦',
    'Switzerland': '🇨🇭',
    'Brazil': '🇧🇷',
    'Morocco': '🇲🇦',
    'Haiti': '🇭🇹',
    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
    'United_States': '🇺🇸',
    'Paraguay': '🇵🇾',
    'Australia': '🇦🇺',
    'Turkey': '🇹🇷',
    'Germany': '🇩🇪',
    'Curacao': '🇨🇼',
    'Ivory_Coast': '🇨🇮',
    'Ecuador': '🇪🇨',
    'Netherlands': '🇳🇱',
    'Japan': '🇯🇵',
    'Sweden': '🇸🇪',
    'Tunisia': '🇹🇳',
    'Belgium': '🇧🇪',
    'Egypt': '🇪🇬',
    'Iran': '🇮🇷',
    'New_Zealand': '🇳🇿',
    'Spain': '🇪🇸',
    'Cape_Verde': '🇨🇻',
    'Saudi_Arabia': '🇸🇦',
    'Uruguay': '🇺🇾',
    'France': '🇫🇷',
    'Senegal': '🇸🇳',
    'Iraq': '🇮🇶',
    'Norway': '🇳🇴',
    'Argentina': '🇦🇷',
    'Algeria': '🇩🇿',
    'Austria': '🇦🇹',
    'Jordan': '🇯🇴',
    'Portugal': '🇵🇹',
    'DR_Congo': '🇨🇩',
    'Uzbekistan': '🇺🇿',
    'Colombia': '🇨🇴',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'Croatia': '🇭🇷',
    'Ghana': '🇬🇭',
    'Panama': '🇵🇦'
}

def format_team_name(team_name: str, include_flag: bool = True) -> str:
    """Formats team name by replacing underscores and optionally prefixing with emoji flags."""
    if not team_name:
        return ""
    # Normalize team name key just in case (e.g. replacing space back to underscore if input is formatted)
    key = team_name.replace(' ', '_')
    display_name = key.replace('_', ' ')
    if not include_flag:
        return display_name
    flag = TEAM_FLAGS.get(key, '🏳️')
    return f"{flag} {display_name}"


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

def estimate_match_ci(p_home, p_draw, p_away, num_sims=2000, num_batches=10):
    """Computes standard error and 95% confidence intervals for match outcome probabilities."""
    runs_per_batch = num_sims // num_batches
    batch_counts = np.random.multinomial(runs_per_batch, [p_home, p_draw, p_away], size=num_batches)
    batch_probs = batch_counts / runs_per_batch
    
    means = np.mean(batch_probs, axis=0)
    stds = np.std(batch_probs, axis=0)
    se = stds / np.sqrt(num_batches)
    
    ci_lower = np.maximum(0.0, means - 1.96 * se)
    ci_upper = np.minimum(1.0, means + 1.96 * se)
    
    return means, ci_lower, ci_upper

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

# Initialize session state for Monte Carlo runs and injury shocks
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None
if 'mc_num_sims' not in st.session_state:
    st.session_state.mc_num_sims = 0
if 'injury_shocks' not in st.session_state:
    st.session_state.injury_shocks = {}

# Sidebar controls
all_teams = sorted(list(sim.starting_elos.keys()))

st.sidebar.markdown("<h3 style='color: #00f2fe; font-weight:700;'>Simulation Parameters</h3>", unsafe_allow_html=True)
num_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=10000, value=2000, step=100)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

run_button = st.sidebar.button("⚡ Run Tournament Simulation", width="stretch")

# Sidebar Injury Shocks
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #00f2fe; font-weight:700;'>🏥 Injury Shock System</h3>", unsafe_allow_html=True)
st.sidebar.write("Simulate rating drops due to key player injuries.")

shock_team = st.sidebar.selectbox("Select Team to Injure", all_teams, key="shock_team_select", format_func=format_team_name)
shock_tier_label = st.sidebar.selectbox(
    "Player Importance",
    ["Key Player (-30 Elo, -0.05 Squad Quality)", "World Class (-50 Elo, -0.10 Squad Quality)", "Indispensable (-80 Elo, -0.15 Squad Quality)"],
    key="shock_tier_select"
)

tier_mapping = {
    "Key Player (-30 Elo, -0.05 Squad Quality)": "key",
    "World Class (-50 Elo, -0.10 Squad Quality)": "world_class",
    "Indispensable (-80 Elo, -0.15 Squad Quality)": "indispensable"
}

if st.sidebar.button("🚨 Apply Injury Shock"):
    st.session_state.injury_shocks[shock_team] = tier_mapping[shock_tier_label]
    sim.apply_injury_shocks(st.session_state.injury_shocks)
    st.session_state.mc_results = sim.run_monte_carlo(num_simulations=num_simulations)

# List active shocks and allow removal
if st.session_state.injury_shocks:
    st.sidebar.markdown("**Active Injury Shocks:**")
    to_remove = []
    for team, tier in list(st.session_state.injury_shocks.items()):
        col_name, col_btn = st.sidebar.columns([3, 1])
        col_name.write(f"⚠️ {format_team_name(team)}: {tier.replace('_', ' ').title()}")
        if col_btn.button("❌", key=f"remove_shock_{team}"):
            to_remove.append(team)
            
    if to_remove:
        for team in to_remove:
            del st.session_state.injury_shocks[team]
        sim.apply_injury_shocks(st.session_state.injury_shocks)
        st.session_state.mc_results = sim.run_monte_carlo(num_simulations=num_simulations)
        st.rerun()

# Run main simulation if requested or not yet run
if run_button or st.session_state.mc_results is None:
    np.random.seed(seed)
    random.seed(seed)
    
    with st.spinner(f"Simulating {num_simulations:,} tournaments..."):
        t0 = time.time()
        sim.apply_injury_shocks(st.session_state.injury_shocks)
        st.session_state.mc_results = sim.run_monte_carlo(num_simulations=num_simulations)
        st.session_state.mc_num_sims = num_simulations
        st.sidebar.success(f"Executed in {time.time() - t0:.2f} seconds!")

# Banner
st.markdown("<div class='header-title'>FIFA 2026 World Cup Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Sports Analytics & Machine Learning Tournament Simulation Dashboard</div>", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Tournament Overview", 
    "⚔️ Match Predictor", 
    "🔍 Team Analysis", 
    "🎮 Scenario Simulator", 
    "⏪ Historical Replay",
    "⚙️ Model Transparency",
    "🧭 Interactive Walkthrough"
])

# ----------------- TAB 1: TOURNAMENT OVERVIEW -----------------
with tab1:
    st.markdown("### 🏆 World Cup 2026 Forecast")
    
    if st.session_state.mc_results:
        probs = st.session_state.mc_results
        
        # Sort teams by champion probability mean
        sorted_probs = sorted(probs.items(), key=lambda x: x[1]['champion']['mean'], reverse=True)
        
        # Create columns
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("<div class='glass-card'><h4>Top 15 Championship Contenders</h4>", unsafe_allow_html=True)
            top_15_teams = [t for t, _ in sorted_probs[:15]]
            top_15_champs = [probs[t]['champion']['mean'] * 100 for t in top_15_teams]
            
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
            st.plotly_chart(fig_champ, width='stretch')
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
                    y=[probs[team][stage]['mean'] * 100 for team in top_10_teams]
                ))
            fig_prog.update_layout(
                barmode='stack',
                template='plotly_dark',
                height=450,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_prog, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Champion DNA Comparison
        st.markdown("<div class='glass-card'><h4>🧬 Champion DNA: Core Metrics Comparison</h4>", unsafe_allow_html=True)
        dna_teams = [t for t, _ in sorted_probs[:5]]
        dna_rows = []
        for team in dna_teams:
            tf = sim.team_features[team]
            coach_info = sim.lookup_system.lookup(team, "2026-06-11")
            dna_rows.append({
                "Team": format_team_name(team),
                "Starting Elo": int(tf['elo']),
                "Squad Quality Score": f"{tf['squad_quality']:.2f}",
                "Club Cohesion": f"{tf['cohesion']:.2%}",
                "Peak Age Ratio (25-29)": f"{tf['peak_age_ratio']:.2%}",
                "Tournament Experience": f"{tf['tournament_experience']:.1f}",
                "Momentum Score": f"{tf['momentum']:.2f}",
                "Coach": coach_info['coach'],
                "Coach Tenure (Years)": f"{coach_info['tenure_days'] / 365.0:.1f}",
                "Championship Prob": f"{probs[team]['champion']['mean']:.2%} [{probs[team]['champion']['ci_lower']:.1%} - {probs[team]['champion']['ci_upper']:.1%}]"
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
        team_a = st.selectbox("Select Team A", all_teams, index=all_teams.index("Brazil"), format_func=format_team_name)
        
        tf_a = sim.team_features[team_a]
        st.metric("Elo Rating", int(tf_a['elo']))
        st.metric("Squad Quality", f"{tf_a['squad_quality']:.2f}")
        
        fatigue_a = st.slider("Team A Fatigue Level", min_value=0, max_value=5, value=0, key="fat_a", help="Heuristic schedule-induced fatigue penalty (-20 Elo per level). This parameter is a simulation heuristic and is not statistically calibrated.")
        st.caption("⚠️ *Fatigue parameters are unvalidated simulation heuristics used to explore match scenarios.*")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 80px; color: #00f2fe;'>VS</h1>", unsafe_allow_html=True)
        
        location_type = st.radio(
            "Match Location",
            options=["Neutral Venue", "Team A Home/Host", "Team B Home/Host"],
            index=0
        )
        
    with col_t2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        team_b = st.selectbox("Select Team B", all_teams, index=all_teams.index("Argentina"), format_func=format_team_name)
        
        tf_b = sim.team_features[team_b]
        st.metric("Elo Rating", int(tf_b['elo']))
        st.metric("Squad Quality", f"{tf_b['squad_quality']:.2f}")
        
        fatigue_b = st.slider("Team B Fatigue Level", min_value=0, max_value=5, value=0, key="fat_b", help="Heuristic schedule-induced fatigue penalty (-20 Elo per level). This parameter is a simulation heuristic and is not statistically calibrated.")
        st.caption("⚠️ *Fatigue parameters are unvalidated simulation heuristics used to explore match scenarios.*")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if team_a == team_b:
        st.warning("Please select two different teams to run a matchup simulation.")
    else:
        # Calculate fatigue and features
        elo_a = tf_a['elo'] - fatigue_a * 20.0
        elo_b = tf_b['elo'] - fatigue_b * 20.0
        
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
            'host_advantage_away': host_advantage_away,
            
            # Additional features for model inputs
            'home_cohesion': tf_a['cohesion'],
            'away_cohesion': tf_b['cohesion'],
            'cohesion_diff': tf_a['cohesion'] - tf_b['cohesion'],
            'home_peak_age_ratio': tf_a['peak_age_ratio'],
            'away_peak_age_ratio': tf_b['peak_age_ratio'],
            'peak_age_ratio_diff': tf_a['peak_age_ratio'] - tf_b['peak_age_ratio'],
            'home_veteran_ratio': tf_a['veteran_ratio'],
            'away_veteran_ratio': tf_b['veteran_ratio'],
            'veteran_ratio_diff': tf_a['veteran_ratio'] - tf_b['veteran_ratio'],
            'home_youth_ratio': tf_a['youth_ratio'],
            'away_youth_ratio': tf_b['youth_ratio'],
            'youth_ratio_diff': tf_a['youth_ratio'] - tf_b['youth_ratio'],
            'home_tournament_experience': tf_a['tournament_experience'],
            'away_tournament_experience': tf_b['tournament_experience'],
            'tournament_experience_diff': tf_a['tournament_experience'] - tf_b['tournament_experience'],
            'home_momentum': tf_a['momentum'],
            'away_momentum': tf_b['momentum'],
            'momentum_diff': tf_a['momentum'] - tf_b['momentum']
        }
        
        df_match = pd.DataFrame([row])[sim.feature_cols]
        base_preds = sim.base_model.predict_proba(df_match)
        cal_preds = sim.calibrator.predict_proba(base_preds)[0]
        
        p_home, p_draw, p_away = cal_preds[0], cal_preds[1], cal_preds[2]
        
        # Estimate Confidence Intervals
        means, ci_low, ci_high = estimate_match_ci(p_home, p_draw, p_away)
        
        st.markdown("<div class='glass-card'><h3>🔮 Calibrated Outcome Probabilities (with 95% Confidence Intervals)</h3>", unsafe_allow_html=True)
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric(f"{format_team_name(team_a)} Win", f"{means[0]:.1%}", help=f"95% CI: [{ci_low[0]:.1%} - {ci_high[0]:.1%}]")
        col_p2.metric("Draw", f"{means[1]:.1%}", help=f"95% CI: [{ci_low[1]:.1%} - {ci_high[1]:.1%}]")
        col_p3.metric(f"{format_team_name(team_b)} Win", f"{means[2]:.1%}", help=f"95% CI: [{ci_low[2]:.1%} - {ci_high[2]:.1%}]")
        
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
        
        st.markdown(f"<h4>Most Likely Scoreline: <span style='color:#00f2fe;'>{format_team_name(team_a)} {top_score[0]} - {top_score[1]} {format_team_name(team_b)}</span> ({top_score_pct:.1%} probability)</h4>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: TEAM ANALYSIS -----------------
with tab3:
    st.markdown("### 🔍 Team Strength & Coach Profile")
    selected_team = st.selectbox("Select a Team for Analysis", all_teams, format_func=format_team_name)
    
    tf_team = sim.team_features[selected_team]
    coach_info = sim.lookup_system.lookup(selected_team, "2026-06-11")
    
    col_ta1, col_ta2 = st.columns(2)
    
    with col_ta1:
        st.markdown("<div class='glass-card'><h4>📈 Strength Profile</h4>", unsafe_allow_html=True)
        st.write(f"**Starting Elo Rating:** {int(tf_team['elo'])}")
        st.write(f"**Squad Quality Score:** {tf_team['squad_quality']:.2f}")
        st.write(f"**Club Cohesion Score:** {tf_team['cohesion']:.2%}")
        st.write(f"**Peak Age Ratio (25-29):** {tf_team['peak_age_ratio']:.2%}")
        st.write(f"**Tournament Experience Index:** {tf_team['tournament_experience']:.1f}")
        st.write(f"**Momentum Index:** {tf_team['momentum']:.2f}")
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
    
    scen_team = st.selectbox("Select Team to Modify", all_teams, key="scen_t", format_func=format_team_name)
    
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
        
    run_scen = st.button("🚀 Run Scenario Simulation (1,000 Runs)", width="stretch")
    
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
        st.markdown("<div class='glass-card'><h3>📊 Scenario Results Comparison (with 95% Confidence Intervals)</h3>", unsafe_allow_html=True)
        
        orig_probs = st.session_state.mc_results[scen_team] if st.session_state.mc_results else None
        new_probs = scen_results[scen_team]
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("<h4>Before Scenario</h4>", unsafe_allow_html=True)
            if orig_probs:
                st.write(f"🏆 **Champion:** {orig_probs['champion']['mean']:.2%} [{orig_probs['champion']['ci_lower']:.1%} - {orig_probs['champion']['ci_upper']:.1%}]")
                st.write(f"🥈 **Runner-up:** {orig_probs['runner_up']['mean']:.2%} [{orig_probs['runner_up']['ci_lower']:.1%} - {orig_probs['runner_up']['ci_upper']:.1%}]")
                st.write(f"🥉 **Third Place:** {orig_probs['third_place']['mean']:.2%} [{orig_probs['third_place']['ci_lower']:.1%} - {orig_probs['third_place']['ci_upper']:.1%}]")
                st.write(f"🚪 **Group Stage Exit:** {orig_probs['group_stage_exit']['mean']:.2%} [{orig_probs['group_stage_exit']['ci_lower']:.1%} - {orig_probs['group_stage_exit']['ci_upper']:.1%}]")
            else:
                st.write("Run the main tournament simulation first to compare.")
                
        with col_res2:
            st.markdown("<h4>After Scenario</h4>", unsafe_allow_html=True)
            st.write(f"🏆 **Champion:** {new_probs['champion']['mean']:.2%} [{new_probs['champion']['ci_lower']:.1%} - {new_probs['champion']['ci_upper']:.1%}]")
            st.write(f"🥈 **Runner-up:** {new_probs['runner_up']['mean']:.2%} [{new_probs['runner_up']['ci_lower']:.1%} - {new_probs['runner_up']['ci_upper']:.1%}]")
            st.write(f"🥉 **Third Place:** {new_probs['third_place']['mean']:.2%} [{new_probs['third_place']['ci_lower']:.1%} - {new_probs['third_place']['ci_upper']:.1%}]")
            st.write(f"🚪 **Group Stage Exit:** {new_probs['group_stage_exit']['mean']:.2%} [{new_probs['group_stage_exit']['ci_lower']:.1%} - {new_probs['group_stage_exit']['ci_upper']:.1%}]")
            
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 5: HISTORICAL REPLAY -----------------
with tab5:
    st.markdown("### ⏪ Historical World Cup Replay & Backtesting")
    st.write("Compare the calibrated model's predictive performance on historical World Cups against actual results.")
    
    if os.path.exists("models/historical_backtests.json"):
        with open("models/historical_backtests.json", "r") as f:
            backtest_data = json.load(f)
            
        selected_wc = st.selectbox("Select Tournament to Replay", ["2022", "2018"])
        
        wc_stats = backtest_data[selected_wc]
        
        # Display overall performance cards
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown(f"""
            <div class='glass-card'>
                <div class='metric-label'>Log Loss on Tournament</div>
                <div class='metric-value'>{wc_stats['overall_log_loss']:.4f}</div>
                <p style='font-size:0.9rem; color:#94a3b8;'>Lower is better. Measures how close the predicted probabilities were to actual outcomes.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_st2:
            st.markdown(f"""
            <div class='glass-card'>
                <div class='metric-label'>Classification Accuracy</div>
                <div class='metric-value'>{wc_stats['overall_accuracy']:.2%}</div>
                <p style='font-size:0.9rem; color:#94a3b8;'>Percentage of match outcomes (Win/Draw/Loss) correctly predicted (argmax).</p>
            </div>
            """, unsafe_allow_html=True)
            
        matches_df = pd.DataFrame(wc_stats["matches"])
        
        # Add display helper columns
        def format_result(row):
            outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            actual_label = outcome_map[row['actual_outcome']]
            pred_probs = [row['probabilities']['home'], row['probabilities']['draw'], row['probabilities']['away']]
            pred_idx = np.argmax(pred_probs)
            pred_label = outcome_map[pred_idx]
            
            status = "✅ Correct" if pred_idx == row['actual_outcome'] else "❌ Incorrect"
            return pred_label, actual_label, status
            
        results = [format_result(r) for _, r in matches_df.iterrows()]
        matches_df["Predicted Outcome"] = [r[0] for r in results]
        matches_df["Actual Outcome"] = [r[1] for r in results]
        matches_df["Status"] = [r[2] for r in results]
        
        matches_df["Home Win Prob"] = matches_df["probabilities"].apply(lambda x: f"{x['home']:.1%}")
        matches_df["Draw Prob"] = matches_df["probabilities"].apply(lambda x: f"{x['draw']:.1%}")
        matches_df["Away Win Prob"] = matches_df["probabilities"].apply(lambda x: f"{x['away']:.1%}")
        
        # Show stage-wise breakdown
        stages_in_tourney = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Third Place Play-off", "Final"]
        for stg in stages_in_tourney:
            stage_matches = matches_df[matches_df["stage"] == stg]
            if not stage_matches.empty:
                with st.expander(f"📅 {stg} ({len(stage_matches)} Matches)"):
                    display_cols = [
                        "date", "home_team", "away_team", "home_score", "away_score",
                        "Home Win Prob", "Draw Prob", "Away Win Prob",
                        "Predicted Outcome", "Actual Outcome", "Status"
                    ]
                    st.dataframe(stage_matches[display_cols].reset_index(drop=True), width='stretch')
    else:
        st.warning("Historical backtest data not found. Please verify the `models/historical_backtests.json` file is present.")

# ----------------- TAB 6: MODEL TRANSPARENCY -----------------
with tab6:
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
            
        st.markdown("""
        <div class='glass-card' style='margin-top: 15px;'>
            <h4>💡 Analysis: 2018 vs. 2022 Calibration Performance Anomaly</h4>
            <p>
                <b>Observation:</b> The 2022 Holdout Set achieves a lower (better) Log Loss (0.9015) 
                and higher Accuracy (58.95%) compared to the 2018 Test Set (Log Loss: 0.9179, Accuracy: 0.5609), 
                despite 2022 being chronologically further from the training window (2000–2013).
            </p>
            <p>
                <b>Key Factors Driving This Anomaly:</b>
                <ul>
                    <li><b>Upset Density and Surprise:</b> The 2018 World Cup had an unusually high frequency of major upsets and draws (e.g., Germany failing to progress from the group stage after losing to South Korea, Argentina struggling, Spain and Argentina exiting early, and Russia's unexpected run). In log loss, assigning a low probability to an outcome that occurs (or a high probability to a favorite that loses) results in a severe cross-entropy penalty.</li>
                    <li><b>Knockout Predictability in 2022:</b> Although the 2022 World Cup had notable group stage upsets (e.g., Argentina losing to Saudi Arabia), the knockout stage was highly structured and aligned closely with Elo and squad value expectations. Major contenders (France, Argentina, Croatia, Brazil) performed consistently, with fewer unexpected knockout draws. This higher alignment of matches with underlying ratings resulted in lower overall Log Loss.</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
            
    # Calibration Metrics Section (Roadmap Point #1)
    st.markdown("### 📈 Platt Scaling Probability Calibration Analysis")
    st.write("Review how Platt Scaling calibrates uncalibrated XGBoost classifier predictions to realistic probabilities.")
    
    if os.path.exists("models/calibration_metrics.json"):
        with open("models/calibration_metrics.json", "r") as f:
            cal_metrics_data = json.load(f)
            
        selected_cal_year = st.radio("Select Evaluation Set", ["2022 World Cup", "2018 World Cup"], horizontal=True)
        key_cal = "holdout_2022" if "2022" in selected_cal_year else "test_2018"
        
        cal_set = cal_metrics_data[key_cal]
        
        col_tbl, col_diag = st.columns([1.2, 1.8])
        
        with col_tbl:
            st.markdown("<div class='glass-card'><h4>Calibration Metrics Comparison</h4>", unsafe_allow_html=True)
            
            tbl_rows = [
                {
                    "Metric": "Log Loss (lower is better)",
                    "Before Calibration": f"{cal_set['before']['log_loss']:.4f}",
                    "After Calibration": f"{cal_set['after']['log_loss']:.4f}"
                },
                {
                    "Metric": "Brier Score (lower is better)",
                    "Before Calibration": f"{cal_set['before']['brier']:.4f}",
                    "After Calibration": f"{cal_set['after']['brier']:.4f}"
                },
                {
                    "Metric": "Expected Calibration Error (ECE)",
                    "Before Calibration": f"{cal_set['before']['ece']:.4f}",
                    "After Calibration": f"{cal_set['after']['ece']:.4f}"
                }
            ]
            st.table(pd.DataFrame(tbl_rows))
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_diag:
            st.markdown("<div class='glass-card'><h4>Reliability Diagram</h4>", unsafe_allow_html=True)
            
            fig_cal = go.Figure()
            
            # Perfect calibration line
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='rgba(255,255,255,0.2)', dash='dash')
            ))
            
            # Uncalibrated curve
            fig_cal.add_trace(go.Scatter(
                x=cal_set['before']['reliability']['bin_confs'],
                y=cal_set['before']['reliability']['bin_accs'],
                mode='lines+markers',
                name='Uncalibrated (XGBoost)',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=6)
            ))
            
            # Calibrated curve
            fig_cal.add_trace(go.Scatter(
                x=cal_set['after']['reliability']['bin_confs'],
                y=cal_set['after']['reliability']['bin_accs'],
                mode='lines+markers',
                name='Calibrated (Platt Scaling)',
                line=dict(color='#00f2fe', width=3),
                marker=dict(size=8)
            ))
            
            fig_cal.update_layout(
                xaxis_title='Mean Predicted Confidence',
                yaxis_title='Actual Observed Accuracy',
                template='plotly_dark',
                height=350,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0)"
                )
            )
            st.plotly_chart(fig_cal, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)
            
    # Feature Importance & Drift in two columns
    col_ta_1, col_ta_2 = st.columns(2)
    
    with col_ta_1:
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
        st.plotly_chart(fig_imp, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_ta_2:
        st.markdown("<div class='glass-card'><h4>🧬 Feature Drift (KS Test Train vs Holdout)</h4>", unsafe_allow_html=True)
        if os.path.exists("models/feature_drift_report.json"):
            with open("models/feature_drift_report.json", "r") as f:
                drift_data = json.load(f)
            drift_rows = []
            for col, d in drift_data.items():
                drift_rows.append({
                    "Feature": col,
                    "Train Mean": f"{d['train_mean']:.3f}",
                    "Holdout Mean": f"{d['holdout_mean']:.3f}",
                    "KS Stat": f"{d['ks_stat']:.3f}",
                    "P-Value": f"{d['p_value']:.4f}",
                    "Drift?": "Yes ⚠️" if d['drift_detected'] else "No"
                })
            st.dataframe(pd.DataFrame(drift_rows).sort_values(by="KS Stat", ascending=False).reset_index(drop=True), width='stretch', height=400)
        else:
            st.write("Feature drift report not found.")
        st.markdown("</div>", unsafe_allow_html=True)
def get_live_group_standings(group_letter, cur_idx, all_104_matches, starting_elos):
    # Get all teams in this group
    teams = GROUPS_2026[group_letter]
    standings = {t: {'points': 0, 'gd': 0, 'gs': 0, 'elo': starting_elos.get(t, 1600.0), 'team': t} for t in teams}
    
    played_matches = []
    # Filter matches in all_104_matches up to cur_idx that belong to this group
    for i in range(min(cur_idx + 1, 72)):
        m = all_104_matches[i]
        if m.get("group") == group_letter:
            t1, t2 = m["team1"], m["team2"]
            g1, g2 = m["goals1"], m["goals2"]
            
            if g1 > g2:
                standings[t1]['points'] += 3
            elif g2 > g1:
                standings[t2]['points'] += 3
            else:
                standings[t1]['points'] += 1
                standings[t2]['points'] += 1
                
            standings[t1]['gd'] += (g1 - g2)
            standings[t1]['gs'] += g1
            standings[t2]['gd'] += (g2 - g1)
            standings[t2]['gs'] += g2
            
            played_matches.append((t1, t2, g1, g2))
            
    # Sort standings using compare_teams
    import functools
    from src.simulator import compare_teams
    sorted_teams = sorted(
        standings.values(),
        key=functools.cmp_to_key(lambda x, y: compare_teams(x, y, played_matches)),
        reverse=True
    )
    return sorted_teams

def get_knockout_stage_status(stage_name, cur_idx, all_104_matches, details):
    # Determine the list of teams in this stage
    if stage_name == "Round of 32":
        teams = details["r32_teams"]
    elif stage_name == "Round of 16":
        teams = details["r16_teams"]
    elif stage_name == "Quarter-finals":
        teams = details["qf_teams"]
    elif stage_name == "Semi-finals":
        teams = details["sf_teams"]
    else:  # Third Place Match or Final
        teams = details["sf_teams"]
        
    status_dict = {t: "⏳ Pending" for t in teams}
    
    # Check matches in this stage up to cur_idx
    for i in range(72, cur_idx + 1):
        m = all_104_matches[i]
        if m.get("stage") == stage_name:
            t1, t2 = m["team1"], m["team2"]
            winner = m["winner"]
            
            if t1 in status_dict:
                status_dict[t1] = "✅ Advanced" if winner == t1 else "❌ Eliminated"
            if t2 in status_dict:
                status_dict[t2] = "✅ Advanced" if winner == t2 else "❌ Eliminated"
                
    # If it is Final or Third Place Match, let's customize
    if stage_name in ["Third Place Match", "Final"]:
        # Find who won/lost the Final and Third Place Match
        for i in range(72, cur_idx + 1):
            m = all_104_matches[i]
            if m.get("stage") == "Third Place Match":
                winner = m["winner"]
                loser = m["team1"] if winner == m["team2"] else m["team2"]
                if winner in status_dict:
                    status_dict[winner] = "🥉 3rd Place"
                if loser in status_dict:
                    status_dict[loser] = "4th Place"
            elif m.get("stage") == "Final":
                winner = m["winner"]
                loser = m["team1"] if winner == m["team2"] else m["team2"]
                if winner in status_dict:
                    status_dict[winner] = "🏆 Champion"
                if loser in status_dict:
                    status_dict[loser] = "🥈 Runner-up"
                    
    return [{"Team": format_team_name(t), "Status": status} for t, status in status_dict.items()]

# ----------------- TAB 7: INTERACTIVE WALKTHROUGH -----------------
with tab7:
    st.markdown("### 🧭 Interactive Tournament Walkthrough")
    st.write(
        "Run a consensus walkthrough grounded in thousands of simulations. "
        "The displayed tournament reflects the **most likely champion** across all runs — "
        "not just a single random draw."
    )

    if 'single_walkthrough' not in st.session_state:
        st.session_state.single_walkthrough = None

    col_btn, col_slider = st.columns([1.5, 2])
    with col_slider:
        walkthrough_sims = st.slider(
            "Simulations to run", min_value=500, max_value=5000,
            value=2000, step=500, key="walkthrough_num_sims",
            help="More runs = more reliable consensus champion. 2,000 is the recommended sweet spot."
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Generate Consensus Walkthrough", key="gen_single_walk"):
            with st.spinner(f"Running {walkthrough_sims:,} simulations to find the consensus outcome..."):
                st.session_state.single_walkthrough = sim.simulate_consensus_walkthrough(
                    num_simulations=walkthrough_sims
                )

    if st.session_state.single_walkthrough:
        res = st.session_state.single_walkthrough
        details = res["details"]
        cs = res.get("consensus_stats")

        # Display champion card
        champ_label = format_team_name(res['champion'])
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-color: rgba(0, 242, 254, 0.4); background: rgba(0, 242, 254, 0.05); padding: 30px;">
            <h2 style="margin: 0; color: #00f2fe; font-size: 2.5rem;">🏆 CHAMPION: {champ_label} 🏆</h2>
            <p style="margin: 10px 0 0 0; color: #94a3b8; font-size: 1.2rem;">
                Runner-up: <b>{format_team_name(res['runner_up'])}</b> | Third Place: <b>{format_team_name(res['third_place'])}</b>
            </p>
            {f'<p style="margin: 8px 0 0 0; color: #4ade80; font-size: 0.95rem;">📊 {champ_label} won in <b>{cs["champion_pct"]:.1%}</b> of {cs["num_simulations"]:,} simulations</p>' if cs else ''}
        </div>
        """, unsafe_allow_html=True)

        # Consensus champion probability table
        if cs:
            with st.expander(f"📊 Champion Probability Table ({cs['num_simulations']:,} simulations)", expanded=False):
                st.markdown("Top 10 most likely champions across all simulation runs:")
                tbl_rows = [
                    {"Rank": i + 1, "Team": format_team_name(r["team"]), "Win Probability": f'{r["pct"]:.1%}'}
                    for i, r in enumerate(cs["top10_champions"])
                ]
                st.table(pd.DataFrame(tbl_rows).set_index("Rank"))

        # Stage Tabs
        w_tab1, w_tab2, w_tab3, w_tab4 = st.tabs([
            "📅 Group Stage",
            "⚖️ Third Place Routing",
            "⚔️ Knockout Bracket",
            "⏱️ Match-by-Match Timeline"
        ])

        with w_tab1:
            st.markdown("#### Group Stage Standings & Fixtures")
            
            # Select Group
            group_letters = list(details["group_matches"].keys())
            selected_group = st.selectbox("Select Group to Inspect", group_letters, format_func=lambda x: f"Group {x}")

            col_matches, col_standings = st.columns([1.5, 1.2])

            with col_matches:
                st.markdown(f"##### Group {selected_group} Fixtures")
                for m in details["group_matches"][selected_group]:
                    t1, t2 = m["team1"], m["team2"]
                    goals1, goals2 = m["goals1"], m["goals2"]
                    probs = m["probs"]  # [Home Win, Draw, Away Win]
                    
                    # Compute prob bar widths
                    p1_w = max(5.0, probs[0] * 100.0)
                    p_draw_w = max(5.0, probs[1] * 100.0)
                    p2_w = max(5.0, probs[2] * 100.0)
                    
                    is_t1_winner = goals1 > goals2
                    is_t2_winner = goals2 > goals1
                    
                    t1_class = "walkthrough-team-winner" if is_t1_winner else ""
                    t2_class = "walkthrough-team-winner" if is_t2_winner else ""
                    card_winner_class = "walkthrough-winner-card" if (is_t1_winner or is_t2_winner) else ""

                    st.markdown(f"""
                    <div class="walkthrough-match-card {card_winner_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="walkthrough-team-name {t1_class}" style="flex: 1;">{format_team_name(t1)}</div>
                            <div class="walkthrough-score" style="width: 80px;">{goals1} - {goals2}</div>
                            <div class="walkthrough-team-name {t2_class}" style="flex: 1; text-align: right;">{format_team_name(t2)}</div>
                        </div>
                        <div class="walkthrough-bar-container">
                            <div class="walkthrough-bar-home" style="width: {p1_w}%;"></div>
                            <div class="walkthrough-bar-draw" style="width: {p_draw_w}%;"></div>
                            <div class="walkthrough-bar-away" style="width: {p2_w}%;"></div>
                        </div>
                        <div class="walkthrough-prob-container">
                            <span>Win: {probs[0]:.1%}</span>
                            <span>Draw: {probs[1]:.1%}</span>
                            <span>Win: {probs[2]:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_standings:
                st.markdown(f"##### Group {selected_group} Standings")
                standings_data = details["group_standings_clean"][selected_group]
                df_standings = pd.DataFrame(standings_data)
                
                # Format/rename columns for display
                df_standings = df_standings.rename(columns={
                    "team": "Team",
                    "points": "Pts",
                    "gd": "GD",
                    "gs": "GS",
                    "elo": "Elo"
                })
                df_standings["Team"] = df_standings["Team"].apply(format_team_name)
                
                # Render using table
                st.dataframe(df_standings.reset_index(drop=True), width=500, height=180)

        with w_tab2:
            st.markdown("#### ⚖️ Best 3rd-Place Teams & Routing Logic")
            st.write(
                "Under the FIFA 2026 format, the 12 third-placed teams are ranked. "
                "The top 8 qualify for the Round of 32. Their matchups depend on "
                "which combination of groups produced the qualified teams."
            )

            routing_data = details["third_place_routing"]
            
            # Left: Standings of 3rd place teams
            # Right: Routing targets
            col_thirds, col_routes = st.columns([1.3, 1.2])

            with col_thirds:
                st.markdown("##### 🥉 Third-Placed Team Rankings")
                
                rows_thirds = []
                best_8_names = set(routing_data["best_eight_thirds"])
                for rank, t_info in enumerate(routing_data["all_third_placed"], 1):
                    team_name = t_info["team"]
                    status = "✅ Qualified" if team_name in best_8_names else "❌ Eliminated"
                    rows_thirds.append({
                        "Rank": rank,
                        "Team": format_team_name(team_name),
                        "Group": t_info["group"],
                        "Pts": t_info["points"],
                        "GD": t_info["gd"],
                        "GS": t_info["gs"],
                        "Elo": int(t_info["elo"]),
                        "Status": status
                    })
                st.table(pd.DataFrame(rows_thirds))

            with col_routes:
                st.markdown("##### 🔗 Round of 32 Opponents Assigned")
                st.write("The 8 qualified teams are assigned to group winners according to the FIFA routing table:")
                
                rows_routes = []
                for winner_slot, third_team in routing_data["routing"].items():
                    rows_routes.append({
                        "Group Winner Slot": f"Winner Group {winner_slot}",
                        "Opponent (3rd Placed)": format_team_name(third_team['team'])
                    })
                st.table(pd.DataFrame(rows_routes))

        with w_tab3:
            st.markdown("#### ⚔️ Knockout Stage Progression")
            
            # Tabs for knockout rounds
            k_tab1, k_tab2, k_tab3, k_tab4, k_tab5 = st.tabs([
                "Round of 32",
                "Round of 16",
                "Quarter-finals",
                "Semi-finals",
                "Finals"
            ])

            def render_knockout_matches(matches_list):
                # Render matches in 2 columns
                cols = st.columns(2)
                for idx, m in enumerate(matches_list):
                    t1, t2 = m["team1"], m["team2"]
                    goals1, goals2 = m["goals1"], m["goals2"]
                    probs = m["probs"]  # [Home, Draw, Away]
                    winner = m["winner"]
                    
                    p1_w = max(5.0, probs[0] * 100.0)
                    p_draw_w = max(5.0, probs[1] * 100.0)
                    p2_w = max(5.0, probs[2] * 100.0)

                    is_t1_winner = winner == t1
                    is_t2_winner = winner == t2

                    t1_class = "walkthrough-team-winner" if is_t1_winner else ""
                    t2_class = "walkthrough-team-winner" if is_t2_winner else ""
                    card_winner_class = "walkthrough-winner-card"

                    badges_html = ""
                    if m.get("penalty_shootout"):
                        p1_shootout_pct = m.get("shootout_p1", 0.5)
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-pk">Pens ({format_team_name(m["shootout_winner"])} won, {p1_shootout_pct:.0%} Prob)</span>'
                    elif m.get("extra_time"):
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-et">Extra Time</span>'

                    # Check if fatigue exists
                    fat_info = ""
                    if m.get("fatigue1", 0) > 0 or m.get("fatigue2", 0) > 0:
                        fat_info = f'<div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Fatigue: {format_team_name(t1)} (F:{m.get("fatigue1", 0)}) vs {format_team_name(t2)} (F:{m.get("fatigue2", 0)})</div>'

                    cols[idx % 2].markdown(f"""
                    <div class="walkthrough-match-card {card_winner_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="walkthrough-team-name {t1_class}" style="flex: 1;">{format_team_name(t1)}</div>
                            <div class="walkthrough-score" style="width: 80px;">{goals1} - {goals2}</div>
                            <div class="walkthrough-team-name {t2_class}" style="flex: 1; text-align: right;">{format_team_name(t2)}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                            {badges_html}
                            <div style="font-size: 0.75rem; color: #94a3b8;">Winner: <b>{format_team_name(winner)}</b></div>
                        </div>
                        {fat_info}
                        <div class="walkthrough-bar-container">
                            <div class="walkthrough-bar-home" style="width: {p1_w}%;"></div>
                            <div class="walkthrough-bar-draw" style="width: {p_draw_w}%;"></div>
                            <div class="walkthrough-bar-away" style="width: {p2_w}%;"></div>
                        </div>
                        <div class="walkthrough-prob-container">
                            <span>Win: {probs[0]:.1%}</span>
                            <span>Draw: {probs[1]:.1%}</span>
                            <span>Win: {probs[2]:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with k_tab1:
                st.markdown("##### Round of 32 Matches")
                render_knockout_matches(details["r32_matches"])

            with k_tab2:
                st.markdown("##### Round of 16 Matches")
                render_knockout_matches(details["r16_matches"])

            with k_tab3:
                st.markdown("##### Quarter-final Matches")
                render_knockout_matches(details["qf_matches"])

            with k_tab4:
                st.markdown("##### Semi-final Matches")
                render_knockout_matches(details["sf_matches"])

            with k_tab5:
                col_final, col_third = st.columns(2)
                with col_final:
                    st.markdown("##### 🏆 World Cup Final")
                    m = details["final_match"][0]
                    t1, t2 = m["team1"], m["team2"]
                    goals1, goals2 = m["goals1"], m["goals2"]
                    probs = m["probs"]
                    winner = m["winner"]
                    
                    p1_w = max(5.0, probs[0] * 100.0)
                    p_draw_w = max(5.0, probs[1] * 100.0)
                    p2_w = max(5.0, probs[2] * 100.0)

                    is_t1_winner = winner == t1
                    is_t2_winner = winner == t2

                    t1_class = "walkthrough-team-winner" if is_t1_winner else ""
                    t2_class = "walkthrough-team-winner" if is_t2_winner else ""

                    badges_html = ""
                    if m.get("penalty_shootout"):
                        p1_shootout_pct = m.get("shootout_p1", 0.5)
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-pk">Pens ({format_team_name(m["shootout_winner"])} won, {p1_shootout_pct:.0%} Prob)</span>'
                    elif m.get("extra_time"):
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-et">Extra Time</span>'

                    st.markdown(f"""
                    <div class="walkthrough-match-card walkthrough-winner-card" style="border-width: 2px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="walkthrough-team-name {t1_class}" style="flex: 1;">{format_team_name(t1)}</div>
                            <div class="walkthrough-score" style="width: 80px; font-size: 1.5rem;">{goals1} - {goals2}</div>
                            <div class="walkthrough-team-name {t2_class}" style="flex: 1; text-align: right;">{format_team_name(t2)}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                            {badges_html}
                            <div style="font-size: 0.8rem; color: #00f2fe; font-weight: bold;">World Champion: {format_team_name(winner)}</div>
                        </div>
                        <div class="walkthrough-bar-container">
                            <div class="walkthrough-bar-home" style="width: {p1_w}%;"></div>
                            <div class="walkthrough-bar-draw" style="width: {p_draw_w}%;"></div>
                            <div class="walkthrough-bar-away" style="width: {p2_w}%;"></div>
                        </div>
                        <div class="walkthrough-prob-container">
                            <span>Win: {probs[0]:.1%}</span>
                            <span>Draw: {probs[1]:.1%}</span>
                            <span>Win: {probs[2]:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_third:
                    st.markdown("##### 🥉 Third Place Match")
                    m = details["third_place_match"][0]
                    t1, t2 = m["team1"], m["team2"]
                    goals1, goals2 = m["goals1"], m["goals2"]
                    probs = m["probs"]
                    winner = m["winner"]
                    
                    p1_w = max(5.0, probs[0] * 100.0)
                    p_draw_w = max(5.0, probs[1] * 100.0)
                    p2_w = max(5.0, probs[2] * 100.0)

                    is_t1_winner = winner == t1
                    is_t2_winner = winner == t2

                    t1_class = "walkthrough-team-winner" if is_t1_winner else ""
                    t2_class = "walkthrough-team-winner" if is_t2_winner else ""

                    badges_html = ""
                    if m.get("penalty_shootout"):
                        p1_shootout_pct = m.get("shootout_p1", 0.5)
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-pk">Pens ({format_team_name(m["shootout_winner"])} won, {p1_shootout_pct:.0%} Prob)</span>'
                    elif m.get("extra_time"):
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-et">Extra Time</span>'

                    st.markdown(f"""
                    <div class="walkthrough-match-card walkthrough-winner-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="walkthrough-team-name {t1_class}" style="flex: 1;">{format_team_name(t1)}</div>
                            <div class="walkthrough-score" style="width: 80px;">{goals1} - {goals2}</div>
                            <div class="walkthrough-team-name {t2_class}" style="flex: 1; text-align: right;">{format_team_name(t2)}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                            {badges_html}
                            <div style="font-size: 0.8rem; color: #94a3b8;">Third Place: <b>{format_team_name(winner)}</b></div>
                        </div>
                        <div class="walkthrough-bar-container">
                            <div class="walkthrough-bar-home" style="width: {p1_w}%;"></div>
                            <div class="walkthrough-bar-draw" style="width: {p_draw_w}%;"></div>
                            <div class="walkthrough-bar-away" style="width: {p2_w}%;"></div>
                        </div>
                        <div class="walkthrough-prob-container">
                            <span>Win: {probs[0]:.1%}</span>
                            <span>Draw: {probs[1]:.1%}</span>
                            <span>Win: {probs[2]:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with w_tab4:
                st.markdown("#### ⏱️ Match-by-Match Tournament Timeline")
                st.write(
                    "Step through all 104 matches of the simulated World Cup in chronological order. "
                    "Observe how group standings update live, and follow the exact progression of qualified teams through the knockouts."
                )
                
                # Construct the flat chronological list of 104 matches
                group_letters = list("ABCDEFGHIJKL")
                group_matches = details["group_matches"]
                
                chronological_group_matches = []
                for g in group_letters:
                    m0 = group_matches[g][0].copy()
                    m0["group"] = g
                    m0["matchday"] = 1
                    m0["match_idx_within_group"] = 0
                    m1 = group_matches[g][1].copy()
                    m1["group"] = g
                    m1["matchday"] = 1
                    m1["match_idx_within_group"] = 1
                    chronological_group_matches.extend([m0, m1])
                    
                for g in group_letters:
                    m2 = group_matches[g][2].copy()
                    m2["group"] = g
                    m2["matchday"] = 2
                    m2["match_idx_within_group"] = 2
                    m3 = group_matches[g][3].copy()
                    m3["group"] = g
                    m3["matchday"] = 2
                    m3["match_idx_within_group"] = 3
                    chronological_group_matches.extend([m2, m3])
                    
                for g in group_letters:
                    m4 = group_matches[g][4].copy()
                    m4["group"] = g
                    m4["matchday"] = 3
                    m4["match_idx_within_group"] = 4
                    m5 = group_matches[g][5].copy()
                    m5["group"] = g
                    m5["matchday"] = 3
                    m5["match_idx_within_group"] = 5
                    chronological_group_matches.extend([m4, m5])
                    
                knockout_matches = []
                for m in details["r32_matches"]:
                    m_copy = m.copy()
                    m_copy["stage"] = "Round of 32"
                    knockout_matches.append(m_copy)
                for m in details["r16_matches"]:
                    m_copy = m.copy()
                    m_copy["stage"] = "Round of 16"
                    knockout_matches.append(m_copy)
                for m in details["qf_matches"]:
                    m_copy = m.copy()
                    m_copy["stage"] = "Quarter-finals"
                    knockout_matches.append(m_copy)
                for m in details["sf_matches"]:
                    m_copy = m.copy()
                    m_copy["stage"] = "Semi-finals"
                    knockout_matches.append(m_copy)
                for m in details["third_place_match"]:
                    m_copy = m.copy()
                    m_copy["stage"] = "Third Place Match"
                    knockout_matches.append(m_copy)
                for m in details["final_match"]:
                    m_copy = m.copy()
                    m_copy["stage"] = "Final"
                    knockout_matches.append(m_copy)
                    
                all_104_matches = chronological_group_matches + knockout_matches
                
                # Step navigation buttons
                if "walkthrough_match_idx" not in st.session_state:
                    st.session_state.walkthrough_match_idx = 0
                    
                col_prev, col_next, col_jump_ko, col_reset = st.columns([1, 1, 1.5, 1])
                with col_prev:
                    if st.button("◀ Previous Match", key="btn_prev_match", disabled=(st.session_state.walkthrough_match_idx == 0)):
                        st.session_state.walkthrough_match_idx = max(0, st.session_state.walkthrough_match_idx - 1)
                        st.rerun()
                with col_next:
                    if st.button("Next Match ▶", key="btn_next_match", disabled=(st.session_state.walkthrough_match_idx == 103)):
                        st.session_state.walkthrough_match_idx = min(103, st.session_state.walkthrough_match_idx + 1)
                        st.rerun()
                with col_jump_ko:
                    if st.button("⏭ Jump to Knockouts", key="btn_jump_ko", disabled=(st.session_state.walkthrough_match_idx >= 72)):
                        st.session_state.walkthrough_match_idx = 72
                        st.rerun()
                with col_reset:
                    if st.button("⏮ Back to Start", key="btn_reset_timeline", disabled=(st.session_state.walkthrough_match_idx == 0)):
                        st.session_state.walkthrough_match_idx = 0
                        st.rerun()
                
                slider_val = st.slider(
                    "Timeline Position (Match 1 to 104)",
                    min_value=1,
                    max_value=104,
                    value=st.session_state.walkthrough_match_idx + 1,
                    key="timeline_slider_control",
                    help="Slide to navigate through the entire tournament in order."
                )
                if slider_val - 1 != st.session_state.walkthrough_match_idx:
                    st.session_state.walkthrough_match_idx = slider_val - 1
                    st.rerun()
                    
                # Current Match Info
                cur_match = all_104_matches[st.session_state.walkthrough_match_idx]
                
                col_left, col_right = st.columns([1.5, 1.2])
                
                with col_left:
                    # Get stage title
                    if st.session_state.walkthrough_match_idx < 72:
                        stage_title = f"Group Stage - Group {cur_match['group']} (Matchday {cur_match['matchday']})"
                    else:
                        stage_title = f"Knockout Stage - {cur_match['stage']}"
                        
                    st.markdown(f"##### 🏟️ Match {st.session_state.walkthrough_match_idx + 1} of 104: {stage_title}")
                    
                    t1, t2 = cur_match["team1"], cur_match["team2"]
                    goals1, goals2 = cur_match["goals1"], cur_match["goals2"]
                    probs = cur_match["probs"]
                    
                    p1_w = max(5.0, probs[0] * 100.0)
                    p_draw_w = max(5.0, probs[1] * 100.0)
                    p2_w = max(5.0, probs[2] * 100.0)
                    
                    if st.session_state.walkthrough_match_idx < 72:
                        is_t1_winner = goals1 > goals2
                        is_t2_winner = goals2 > goals1
                    else:
                        winner = cur_match["winner"]
                        is_t1_winner = winner == t1
                        is_t2_winner = winner == t2
                        
                    t1_class = "walkthrough-team-winner" if is_t1_winner else ""
                    t2_class = "walkthrough-team-winner" if is_t2_winner else ""
                    card_winner_class = "walkthrough-winner-card"
                    
                    badges_html = ""
                    if cur_match.get("penalty_shootout"):
                        p1_shootout_pct = cur_match.get("shootout_p1", 0.5)
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-pk">Pens ({format_team_name(cur_match["shootout_winner"])} won, {p1_shootout_pct:.0%} Prob)</span>'
                    elif cur_match.get("extra_time"):
                        badges_html = f'<span class="walkthrough-badge walkthrough-badge-et">Extra Time</span>'
                        
                    fat_info = ""
                    if cur_match.get("fatigue1", 0) > 0 or cur_match.get("fatigue2", 0) > 0:
                        fat_info = f'<div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Fatigue: {format_team_name(t1)} (F:{cur_match.get("fatigue1", 0)}) vs {format_team_name(t2)} (F:{cur_match.get("fatigue2", 0)})</div>'
                        
                    st.markdown(f"""
                    <div class="walkthrough-match-card {card_winner_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="walkthrough-team-name {t1_class}" style="flex: 1;">{format_team_name(t1)}</div>
                            <div class="walkthrough-score" style="width: 80px; font-size: 1.3rem;">{goals1} - {goals2}</div>
                            <div class="walkthrough-team-name {t2_class}" style="flex: 1; text-align: right;">{format_team_name(t2)}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                            {badges_html}
                            <div style="font-size: 0.75rem; color: #94a3b8;">
                                {"Winner: <b>" + format_team_name(cur_match["winner"]) + "</b>" if st.session_state.walkthrough_match_idx >= 72 else ""}
                            </div>
                        </div>
                        {fat_info}
                        <div class="walkthrough-bar-container">
                            <div class="walkthrough-bar-home" style="width: {p1_w}%;"></div>
                            <div class="walkthrough-bar-draw" style="width: {p_draw_w}%;"></div>
                            <div class="walkthrough-bar-away" style="width: {p2_w}%;"></div>
                        </div>
                        <div class="walkthrough-prob-container">
                            <span>Win: {probs[0]:.1%}</span>
                            <span>Draw: {probs[1]:.1%}</span>
                            <span>Win: {probs[2]:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Render mini match details/stats breakdown
                    st.markdown("###### Pre-Match Probability Distribution")
                    st.write(
                        f"Our model estimated a **{probs[0]:.1%}** chance of a win for {format_team_name(t1)}, "
                        f"a **{probs[1]:.1%}** chance of a draw, and a **{probs[2]:.1%}** chance of a win for {format_team_name(t2)}."
                    )
                    
                with col_right:
                    if st.session_state.walkthrough_match_idx < 72:
                        # Group stage: show live group standings
                        g_letter = cur_match["group"]
                        st.markdown(f"##### 📊 Live Standings - Group {g_letter}")
                        st.write("Calculated live up to and including the current match.")
                        
                        live_standings = get_live_group_standings(g_letter, st.session_state.walkthrough_match_idx, all_104_matches, sim.starting_elos)
                        df_live = pd.DataFrame(live_standings)
                        df_live = df_live.rename(columns={
                            "team": "Team",
                            "points": "Pts",
                            "gd": "GD",
                            "gs": "GS",
                            "elo": "Elo"
                        })
                        df_live["Team"] = df_live["Team"].apply(format_team_name)
                        st.dataframe(df_live.reset_index(drop=True), width=500, height=180)
                    else:
                        # Knockout stage: show qualified teams list
                        stage_name = cur_match["stage"]
                        st.markdown(f"##### ⚔️ {stage_name} Live Status")
                        st.write("Real-time progression status of teams in the current stage.")
                        
                        status_list = get_knockout_stage_status(stage_name, st.session_state.walkthrough_match_idx, all_104_matches, details)
                        df_status = pd.DataFrame(status_list)
                        st.dataframe(df_status.reset_index(drop=True), width=500, height=350)
    else:
        st.info("💡 Click the button above to run a single World Cup 2026 simulation and view the match-by-match details!")
