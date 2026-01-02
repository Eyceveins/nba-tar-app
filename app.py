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
    numeric_cols = df.columns.drop([c for c in non_numeric if c in df.columns])
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
    numeric_cols = df.columns.drop([c for c in non_numeric if c in df.columns])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

# -----------------------------
# TAR Calculation
# -----------------------------
def calculate_tar(player, year):
    poss = get_season_stats(year)
    adv = get_advanced_stats(year)

    poss["Player_clean"] = poss["Player"].apply(clean_player_name)
    adv["Player_clean"] = adv["Player"].apply(clean_player_name)

    df = poss.merge(adv, on="Player_clean", how="left", suffixes=("_poss", "_adv"))

    rename_map = {
        "MP_poss": "MP",
        "PTS_poss": "PTS",
        "AST_poss": "AST",
        "ORB_poss": "ORB",
        "DRB_poss": "DRB",
        "TOV_poss": "TOV",
        "STL_poss": "STL",
        "BLK_poss": "BLK",
        "TS%_adv": "TS%",
        "DRtg_adv": "DRtg",
        "3PAr_adv": "3PAr",
        "FTr_adv": "FTr"
    }
    df.rename(columns=rename_map, inplace=True)

    player_cleaned = clean_player_name(player)
    if player_cleaned not in df["Player_clean"].values:
        raise ValueError(f"Player '{player}' not found for {year} season.")

    p = df[df["Player_clean"] == player_cleaned].iloc[0]

    pos_col = "Pos_poss" if "Pos_poss" in df.columns else "Pos"
    pos = p[pos_col]

    pos_filter = df[pos_col] == pos

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
    threepar_avg = league_avg(df[pos_filter]["3PAr"]) if "3PAr" in df.columns else 0.2
    ftr_avg = league_avg(df[pos_filter]["FTr"]) if "FTr" in df.columns else 0.2

    # -----------------------------
    # Offensive Rating (AOR)
    # -----------------------------
    ts_factor = p["TS%"] / ts_avg
    scoring_factor = p["PTS"] / pts_avg
    orb_factor = p["ORB"] / orb_avg
    tov_factor = tov_avg / p["TOV"] if p["TOV"] > 0 else 1
    creation_factor = p["AST"] / ast_avg

    shooting_factor = 1 + (
        (p.get("3PAr", 0) / threepar_avg +
         p.get("FTr", 0) / ftr_avg) / 2
    )

    AOR = (
        0.25 * ts_factor +
        0.25 * scoring_factor +
        0.15 * orb_factor +
        0.15 * tov_factor +
        0.20 * creation_factor
    ) * shooting_factor

    # -----------------------------
    # Defensive Rating (ADR)
    # -----------------------------
    drtg_factor = math.sqrt(drtg_avg / p["DRtg"])
    drb_factor = min(p["DRB"] / drb_avg, 1.6)
    stl_factor = min(p["STL"] / stl_avg, 1.6)
    blk_factor = min(p["BLK"] / blk_avg, 1.6)

    if pos in ["PG", "SG"]:
        ADR = (
            0.45 * drtg_factor +
            0.35 * stl_factor +
            0.15 * drb_factor +
            0.05 * blk_factor
        )
    elif pos == "SF":
        ADR = (
            0.40 * drtg_factor +
            0.25 * drb_factor +
            0.20 * stl_factor +
            0.15 * blk_factor
        )
    else:
        ADR = (
            0.35 * drtg_factor +
            0.30 * drb_factor +
            0.10 * stl_factor +
            0.25 * blk_factor
        )

    minute_factor = min(1.0, p["MP"] / mp_avg)
    TAR = AOR * ADR * minute_factor

    return {
        "AOR": round(AOR, 3),
        "ADR": round(ADR, 3),
        "TAR": round(TAR, 3),
        "MP": round(p["MP"], 1),
        "ShootingFactor": round(shooting_factor, 3)
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NBA TAR Comparison", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0D1117; color: white; }
    .stTextInput>div>input, .stNumberInput>div>input {
        background-color: #1E1E1E;
        color: white;
        border: 1px solid #444;
    }
    .stButton>button {
        background-color: #FF4500;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏀 NBA TAR Player Comparison")
st.write("Compare players across seasons using Total Adjusted Rating (TAR).")

col1, col2 = st.columns(2)
with col1:
    player_a = st.text_input("Player A")
    year_a = st.number_input("Season A", 1950, 2025, 2016)

with col2:
    player_b = st.text_input("Player B")
    year_b = st.number_input("Season B", 1950, 2025, 2024)

if st.button("Compare Players"):
    try:
        a = calculate_tar(player_a, year_a)
        b = calculate_tar(player_b, year_b)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"{player_a} ({year_a})")
            st.write(a)
        with c2:
            st.subheader(f"{player_b} ({year_b})")
            st.write(b)

        if a["TAR"] > b["TAR"]:
            st.success(f"🏆 {player_a} wins")
        elif b["TAR"] > a["TAR"]:
            st.success(f"🏆 {player_b} wins")
        else:
            st.info("🤝 Tie")

    except Exception as e:
        st.error(str(e))
