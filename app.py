import streamlit as st
import pandas as pd
import math
import re

# -----------------------------
# Helper functions
# -----------------------------
def clean_player_name(name):
    return re.sub(r'[^A-Za-z]', '', str(name)).lower()

def league_avg(series):
    return series.mean(skipna=True)

def clean_dataframe(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns]
    df.columns = df.columns.str.strip().str.replace('\n', '', regex=False)
    return df

# -----------------------------
# Data fetching
# -----------------------------
@st.cache_data(show_spinner=False)
def get_season_stats(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html"
    df = pd.read_html(url)[0]
    df = clean_dataframe(df)
    df = df[df["Player"] != "Player"]

    team_col = next((c for c in ["Tm", "Team", "team"] if c in df.columns), None)
    if team_col:
        df.rename(columns={team_col: "Tm"}, inplace=True)

    non_numeric = ["Player", "Pos", "Tm"]
    numeric_cols = df.columns.drop([col for col in non_numeric if col in df.columns])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def get_advanced_stats(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    df = pd.read_html(url)[0]
    df = clean_dataframe(df)
    df = df[df["Player"] != "Player"]

    team_col = next((c for c in ["Tm", "Team", "team"] if c in df.columns), None)
    if team_col:
        df.rename(columns={team_col: "Tm"}, inplace=True)

    non_numeric = ["Player", "Pos", "Tm"]
    numeric_cols = df.columns.drop([col for col in non_numeric if col in df.columns])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

# -----------------------------
# TAR Calculation
# -----------------------------
def calculate_tar(player, year):
    poss = get_season_stats(year)
    adv = get_advanced_stats(year)

    # Clean player names
    poss['Player_clean'] = poss['Player'].apply(clean_player_name)
    adv['Player_clean'] = adv['Player'].apply(clean_player_name)

    # Merge datasets
    df = poss.merge(adv, on="Player_clean", how="left", suffixes=('_poss', '_adv'))

    # Rename merged columns to standard names
    rename_map = {
        'MP_poss': 'MP',
        'PTS_poss': 'PTS',
        'AST_poss': 'AST',
        'ORB_poss': 'ORB',
        'DRB_poss': 'DRB',
        'TOV_poss': 'TOV',
        'STL_poss': 'STL',
        'BLK_poss': 'BLK',
        'TS%_adv': 'TS%',
        'DRtg_adv': 'DRtg',
        '3PAr_adv': '3PAr',
        'FTr_adv': 'FTr'
    }
    df.rename(columns=rename_map, inplace=True)

    # Check player exists
    player_cleaned = clean_player_name(player)
    if player_cleaned not in df["Player_clean"].values:
        raise ValueError(f"Player '{player}' not found for {year} season.")

    p = df[df["Player_clean"] == player_cleaned].iloc[0]

    # Position-based league averages
    pos_filter = df['Pos_poss'] == p['Pos_poss'] if 'Pos_poss' in df.columns else df['Pos'] == p['Pos']
    ts_avg = league_avg(df[pos_filter]["TS%"])
    pts_avg = league_avg(df[pos_filter]["PTS"])
    orb_avg = league_avg(df[pos_filter]["ORB"])
    tov_avg = league_avg(df[pos_filter]["TOV"])
    ast_avg = league_avg(df[pos_filter]["AST"])
    drb_avg = league_avg(df[pos_filter]["DRB"])
    stl_avg = league_avg(df[pos_filter]["STL"])
    blk_avg = league_avg(df[pos_filter]["BLK"])
    drtg_avg = league_avg(df[pos_filter]["DRtg"])
    mp_avg = league_avg(df[pos_filter]["MP"])
    threepar_avg = league_avg(df[pos_filter]["3PAr"]) if '3PAr' in df.columns else 0.2
    ftr_avg = league_avg(df[pos_filter]["FTr"]) if 'FTr' in df.columns else 0.2

    # -----------------------------
    # Offensive factors
    # -----------------------------
    ts_factor = p["TS%"] / ts_avg
    scoring_factor = p["PTS"] / pts_avg
    orb_factor = p["ORB"] / orb_avg
    tov_factor = tov_avg / p["TOV"] if p["TOV"] > 0 else 1
    creation_factor = p["AST"] / ast_avg

    # Corrected shooting factor (relative to league avg)
    shooting_factor = math.sqrt((p.get("3PAr",0)/threepar_avg) * (p.get("FTr",0)/ftr_avg))
    shooting_factor = max(0.85, min(shooting_factor, 1.25))

    # Weighted additive AOR
    AOR = 0.25*ts_factor + 0.25*scoring_factor + 0.15*orb_factor + 0.15*tov_factor + 0.20*creation_factor
    AOR *= shooting_factor

    # -----------------------------
    # Defensive factors (role-aware)
    # -----------------------------
    drtg_factor = drtg_avg / p["DRtg"]
    drb_factor = p["DRB"] / drb_avg
    stl_factor = p["STL"] / stl_avg
    blk_factor = p["BLK"] / blk_avg

    # Compress extremes
    stl_factor = min(stl_factor, 1.6)
    blk_factor = min(blk_factor, 1.6)
    drb_factor = min(drb_factor, 1.6)
    drtg_factor = math.sqrt(drtg_factor)

    pos = p["Pos_poss"]
    if pos in ["PG","SG"]:
        ADR = 0.45*drtg_factor + 0.35*stl_factor + 0.15*drb_factor + 0.05*blk_factor
    elif pos == "SF":
        ADR = 0.40*drtg_factor + 0.25*drb_factor + 0.20*stl_factor + 0.15*blk_factor
    else:
        ADR = 0.35*drtg_factor + 0.30*drb_factor + 0.10*stl_factor + 0.25*blk_factor

    # Minute factor
    minute_factor = min(1.0, p["MP"]/mp_avg)

    # Final TAR
    TAR = AOR * ADR * minute_factor

    return {
        "AOR": round(AOR,3),
        "ADR": round(ADR,3),
        "TAR": round(TAR,3),
        "MP": round(p["MP"],1),
        "ShootingFactor": round(shooting_factor,3)
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NBA TAR Comparison", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0D1117; color: white; }
.stTextInput>div>input, .stNumberInput>div>input { background-color: #1E1E1E; color: white; border: 1px solid #444; }
.stButton>button { background-color: #FF4500; color: white; font-weight: bold; }
.css-1d391kg {color:white;}
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA TAR Player Comparison")
st.write("Compare players across seasons with Total Adjusted Rating (TAR). Weighted additive formula with shooting boost.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Player A")
    player_a = st.text_input("Player Name", key="player_a")
    year_a = st.number_input("Season Year", 1950, 2025, 2016, key="year_a")

with col2:
    st.subheader("Player B")
    player_b = st.text_input("Player Name", key="player_b")
    year_b = st.number_input("Season Year", 1950, 2025, 2024, key="year_b")

if st.button("Compare Players"):
    try:
        result_a = calculate_tar(player_a, year_a)
        result_b = calculate_tar(player_b, year_b)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {player_a} ({year_a})")
            st.write(result_a)
        with c2:
            st.markdown(f"### {player_b} ({year_b})")
            st.write(result_b)

        st.divider()
        if result_a["TAR"] > result_b["TAR"]:
            st.success(f"🏆 {player_a} ({year_a}) has the higher TAR")
        elif result_b["TAR"] > result_a["TAR"]:
            st.success(f"🏆 {player_b} ({year_b}) has the higher TAR")
        else:
            st.info("🤝 The players are evenly matched by TAR")

    except Exception as e:
        st.error(str(e))
