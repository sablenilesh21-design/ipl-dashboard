import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }

  .main { background: #0a0e1a; }
  .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 100%); }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2a45 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s;
  }
  .kpi-card:hover { transform: translateY(-3px); }
  .kpi-title { color: #8899bb; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
  .kpi-value { color: #f0a500; font-family: 'Rajdhani', sans-serif; font-size: 32px; font-weight: 700; line-height: 1; }
  .kpi-sub { color: #aabbcc; font-size: 11px; margin-top: 6px; }

  /* Section headers */
  .section-header {
    background: linear-gradient(90deg, #f0a500 0%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Rajdhani', sans-serif;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 10px 0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111e35 100%);
    border-right: 1px solid #1e2d45;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label { color: #aabbcc !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #1a2035; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #8899bb; font-family: 'Rajdhani', sans-serif; font-size: 15px; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: #f0a500 !important; color: #000 !important; border-radius: 6px; }

  /* Tables */
  .dataframe { background: #1a2035 !important; color: #ccd6f6 !important; }

  /* Divider */
  hr { border-color: #1e2d45; }

  /* Prediction box */
  .pred-box {
    background: linear-gradient(135deg, #1a2035, #1e2a45);
    border-radius: 14px;
    padding: 24px;
    border: 1px solid #2a3a5c;
    margin: 10px 0;
  }
  .win-prob-bar {
    height: 28px;
    border-radius: 6px;
    background: linear-gradient(90deg, #f0a500, #ff6b35);
    display: flex;
    align-items: center;
    padding: 0 10px;
    font-weight: 700;
    color: #000;
    font-size: 14px;
  }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    # Normalize team names
    name_map = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
    }
    for col in ['team1', 'team2', 'winner', 'toss_winner']:
        matches[col] = matches[col].replace(name_map)
    for col in ['batting_team', 'bowling_team']:
        deliveries[col] = deliveries[col].replace(name_map)

    return matches, deliveries

matches, deliveries = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px'>
      <div style='font-family:Rajdhani; font-size:28px; font-weight:700;
                  background:linear-gradient(90deg,#f0a500,#ff6b35);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🏏 IPL ANALYTICS
      </div>
      <div style='color:#556688; font-size:11px; letter-spacing:2px;'>2008 – 2024 · 17 SEASONS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    all_seasons = ["All Seasons"] + sorted(matches["season"].unique().tolist())
    selected_season = st.selectbox("📅 Season", all_seasons)

    all_teams = ["All Teams"] + sorted(matches["team1"].unique().tolist())
    selected_team = st.selectbox("🏟️ Team", all_teams)

    st.markdown("---")
    st.markdown("<div style='color:#556688;font-size:11px;text-align:center'>Data: Kaggle IPL Dataset<br>1095 matches · 260K+ deliveries</div>", unsafe_allow_html=True)

# ── FILTER DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def filter_data(season, team):
    m = matches.copy()
    d = deliveries.copy()
    if season != "All Seasons":
        m = m[m["season"] == season]
        d = d[d["match_id"].isin(m["id"])]
    if team != "All Teams":
        m = m[(m["team1"] == team) | (m["team2"] == team)]
        d = d[d["match_id"].isin(m["id"])]
    return m, d

fmatches, fdel = filter_data(selected_season, selected_team)

# ── PRECOMPUTE STATS ──────────────────────────────────────────────────────────
@st.cache_data
def compute_batting(d):
    bat = d.groupby("batter").agg(
        runs=("batsman_runs", "sum"),
        balls=("batsman_runs", "count"),
        fours=("batsman_runs", lambda x: (x == 4).sum()),
        sixes=("batsman_runs", lambda x: (x == 6).sum()),
    ).reset_index()
    inn = d.groupby(["match_id", "batter"])["batsman_runs"].sum().reset_index()
    inn.columns = ["match_id", "batter", "score"]
    bat = bat.merge(inn.groupby("batter").size().reset_index(name="innings"), on="batter")
    bat = bat.merge(inn.groupby("batter")["score"].max().reset_index(name="highest"), on="batter")
    diss = d[d["is_wicket"] == 1].groupby("player_dismissed").size().reset_index(name="outs")
    bat = bat.merge(diss, left_on="batter", right_on="player_dismissed", how="left").drop(columns="player_dismissed", errors="ignore")
    bat["outs"] = bat["outs"].fillna(0)
    bat["avg"] = (bat["runs"] / bat["outs"].replace(0, np.nan)).round(2)
    bat["sr"] = ((bat["runs"] / bat["balls"]) * 100).round(2)
    return bat.sort_values("runs", ascending=False)

@st.cache_data
def compute_bowling(d):
    bowl = d.groupby("bowler").agg(
        balls=("total_runs", "count"),
        runs_given=("total_runs", "sum"),
        wickets=("is_wicket", "sum")
    ).reset_index()
    bowl["econ"] = ((bowl["runs_given"] / bowl["balls"]) * 6).round(2)
    bowl["avg"] = (bowl["runs_given"] / bowl["wickets"].replace(0, np.nan)).round(2)
    bowl["sr"] = (bowl["balls"] / bowl["wickets"].replace(0, np.nan)).round(2)
    return bowl.sort_values("wickets", ascending=False)

bat_df = compute_batting(fdel)
bowl_df = compute_bowling(fdel)

# ── HELPER: KPI card ─────────────────────────────────────────────────────────
def kpi(title, value, sub=""):
    return f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
    </div>"""

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOT_BG = "#0d1526"
PAPER_BG = "#0d1526"
FONT_COLOR = "#aabbcc"
GRID_COLOR = "#1e2d45"
COLORS = ["#f0a500", "#ff6b35", "#4ecdc4", "#45b7d1", "#96ceb4",
          "#ffeaa7", "#fd79a8", "#6c5ce7", "#00b894", "#e17055"]

def plot_layout(fig, title="", height=380):
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font_color=FONT_COLOR, font_family="Inter",
        title=dict(text=title, font=dict(family="Rajdhani", size=18, color="#f0a500")),
        height=height, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont_color=FONT_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont_color=FONT_COLOR),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:10px 0 5px'>
  <div style='font-family:Rajdhani; font-size:42px; font-weight:700;
              background:linear-gradient(90deg,#f0a500 0%,#ff6b35 100%);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              letter-spacing:3px;'>
    🏏 IPL ANALYTICS DASHBOARD
  </div>
  <div style='color:#556688; letter-spacing:3px; font-size:12px; margin-top:4px;'>
    INDIAN PREMIER LEAGUE · COMPLETE STATISTICAL ANALYSIS
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TOP KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
total_runs = int(fdel["batsman_runs"].sum())
total_wickets = int(fdel["is_wicket"].sum())
total_sixes = int((fdel["batsman_runs"] == 6).sum())
total_fours = int((fdel["batsman_runs"] == 4).sum())
orange_cap = bat_df.iloc[0]["batter"] if len(bat_df) > 0 else "N/A"
orange_runs = int(bat_df.iloc[0]["runs"]) if len(bat_df) > 0 else 0
purple_cap = bowl_df.iloc[0]["bowler"] if len(bowl_df) > 0 else "N/A"
purple_wkts = int(bowl_df.iloc[0]["wickets"]) if len(bowl_df) > 0 else 0

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.markdown(kpi("Matches", f"{len(fmatches):,}"), unsafe_allow_html=True)
c2.markdown(kpi("Total Runs", f"{total_runs:,}"), unsafe_allow_html=True)
c3.markdown(kpi("Wickets", f"{total_wickets:,}"), unsafe_allow_html=True)
c4.markdown(kpi("Sixes 💥", f"{total_sixes:,}"), unsafe_allow_html=True)
c5.markdown(kpi("Fours", f"{total_fours:,}"), unsafe_allow_html=True)
c6.markdown(kpi("🟠 Orange Cap", orange_cap, f"{orange_runs} runs"), unsafe_allow_html=True)
c7.markdown(kpi("🟣 Purple Cap", purple_cap, f"{purple_wkts} wkts"), unsafe_allow_html=True)

# Avg first innings
fi_avg = fdel[fdel["inning"] == 1].groupby("match_id")["total_runs"].sum().mean()
c8.markdown(kpi("Avg 1st Inn.", f"{fi_avg:.0f}"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["🏏 Batting", "🎳 Bowling", "🏆 Teams", "🏟️ Venues & Matches", "🤖 Win Predictor"])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — BATTING                                             ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[0]:
    st.markdown('<p class="section-header">TOP BATSMEN ANALYSIS</p>', unsafe_allow_html=True)

    top10 = bat_df.head(10).copy()
    top10["avg"] = top10["avg"].fillna(0)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Bar chart – Top Run Scorers
        fig = go.Figure(go.Bar(
            x=top10["runs"], y=top10["batter"],
            orientation="h",
            marker=dict(
                color=top10["runs"],
                colorscale=[[0, "#ff6b35"], [1, "#f0a500"]],
                showscale=False,
            ),
            text=top10["runs"].apply(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color="#f0a500", size=11),
            hovertemplate="<b>%{y}</b><br>Runs: %{x:,}<extra></extra>"
        ))
        plot_layout(fig, "Top 10 Run Scorers (All Time)", height=400)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie – Runs by Team
        tr = fdel.groupby("batting_team")["batsman_runs"].sum().reset_index()
        tr.columns = ["team", "runs"]
        tr = tr.sort_values("runs", ascending=False).head(10)
        fig2 = go.Figure(go.Pie(
            labels=tr["team"], values=tr["runs"],
            hole=0.45,
            marker=dict(colors=COLORS),
            textinfo="label+percent",
            textfont=dict(size=10, color=FONT_COLOR),
            hovertemplate="<b>%{label}</b><br>Runs: %{value:,}<br>%{percent}<extra></extra>"
        ))
        plot_layout(fig2, "Runs Distribution by Team", height=400)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Scatter – SR vs Avg (min 500 runs)
        scatter_df = bat_df[bat_df["runs"] >= 500].copy().fillna(0)
        fig3 = px.scatter(
            scatter_df, x="avg", y="sr",
            size="runs", color="sixes",
            hover_name="batter",
            hover_data={"runs": True, "avg": True, "sr": True, "sixes": True},
            color_continuous_scale=[[0,"#4ecdc4"],[1,"#f0a500"]],
            size_max=30,
        )
        plot_layout(fig3, "Strike Rate vs Average (≥500 runs)", height=380)
        fig3.update_layout(coloraxis_colorbar=dict(title="Sixes", tickfont_color=FONT_COLOR))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Bar – Fours & Sixes
        top8 = bat_df.head(8)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Fours", x=top8["batter"], y=top8["fours"],
                              marker_color="#45b7d1",
                              hovertemplate="<b>%{x}</b><br>Fours: %{y}<extra></extra>"))
        fig4.add_trace(go.Bar(name="Sixes", x=top8["batter"], y=top8["sixes"],
                              marker_color="#f0a500",
                              hovertemplate="<b>%{x}</b><br>Sixes: %{y}<extra></extra>"))
        fig4.update_layout(barmode="group")
        plot_layout(fig4, "Fours & Sixes – Top 8 Batters", height=380)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### 📊 Detailed Batting Leaderboard")
    display_bat = bat_df.head(20)[["batter", "runs", "innings", "avg", "sr", "fours", "sixes", "highest"]].copy()
    display_bat.columns = ["Batter", "Runs", "Innings", "Average", "Strike Rate", "4s", "6s", "Highest"]
    display_bat.index = range(1, len(display_bat)+1)
    st.dataframe(display_bat, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — BOWLING                                             ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[1]:
    st.markdown('<p class="section-header">TOP BOWLERS ANALYSIS</p>', unsafe_allow_html=True)

    top10b = bowl_df[bowl_df["wickets"] >= 5].head(10).copy()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=top10b["wickets"], y=top10b["bowler"],
            orientation="h",
            marker=dict(color=top10b["wickets"],
                        colorscale=[[0, "#6c5ce7"], [1, "#fd79a8"]]),
            text=top10b["wickets"],
            textposition="outside", textfont=dict(color="#fd79a8", size=12),
            hovertemplate="<b>%{y}</b><br>Wickets: %{x}<extra></extra>"
        ))
        plot_layout(fig, "Top 10 Wicket Takers", height=400)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Economy distribution
        econ_df = bowl_df[(bowl_df["balls"] >= 120) & (bowl_df["econ"] < 15)].copy()
        fig2 = px.histogram(
            econ_df, x="econ", nbins=30,
            color_discrete_sequence=["#4ecdc4"]
        )
        fig2.add_vline(x=econ_df["econ"].mean(), line_color="#f0a500",
                       annotation_text=f"Avg: {econ_df['econ'].mean():.2f}",
                       annotation_font_color="#f0a500")
        plot_layout(fig2, "Economy Rate Distribution (≥20 overs bowled)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Scatter – Economy vs Wickets
        scat = bowl_df[(bowl_df["wickets"] >= 20) & (bowl_df["econ"] < 15)].copy().fillna(0)
        fig3 = px.scatter(
            scat, x="econ", y="wickets",
            size="balls", hover_name="bowler",
            color="avg",
            color_continuous_scale=[[0,"#00b894"],[1,"#e17055"]],
            size_max=25,
            hover_data={"econ": True, "wickets": True, "avg": True}
        )
        plot_layout(fig3, "Economy vs Wickets (≥20 wickets)", height=380)
        fig3.update_layout(coloraxis_colorbar=dict(title="Avg", tickfont_color=FONT_COLOR))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Bowling avg top 10
        top_bowl_avg = bowl_df[(bowl_df["wickets"] >= 30)].nsmallest(10, "avg").copy().fillna(0)
        fig4 = go.Figure(go.Bar(
            x=top_bowl_avg["bowler"], y=top_bowl_avg["avg"],
            marker=dict(color=top_bowl_avg["avg"],
                        colorscale=[[0,"#00b894"],[1,"#e17055"]]),
            text=top_bowl_avg["avg"].round(1),
            textposition="outside", textfont_color="#aabbcc",
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f}<extra></extra>"
        ))
        plot_layout(fig4, "Best Bowling Average (≥30 wickets)", height=380)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### 📊 Detailed Bowling Leaderboard")
    display_bowl = bowl_df.head(20)[["bowler","wickets","balls","runs_given","econ","avg","sr"]].copy()
    display_bowl["overs"] = (display_bowl["balls"] // 6).astype(str) + "." + (display_bowl["balls"] % 6).astype(str)
    display_bowl = display_bowl[["bowler","wickets","overs","runs_given","econ","avg","sr"]]
    display_bowl.columns = ["Bowler","Wickets","Overs","Runs Given","Economy","Average","Strike Rate"]
    display_bowl.index = range(1, len(display_bowl)+1)
    st.dataframe(display_bowl, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — TEAMS                                               ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[2]:
    st.markdown('<p class="section-header">TEAM PERFORMANCE ANALYSIS</p>', unsafe_allow_html=True)

    @st.cache_data
    def team_stats_df(m):
        teams = sorted(set(m["team1"].tolist() + m["team2"].tolist()))
        rows = []
        for t in teams:
            played = m[(m["team1"]==t)|(m["team2"]==t)]
            wins = (played["winner"]==t).sum()
            toss_w = (played["toss_winner"]==t).sum()
            toss_match_w = ((played["toss_winner"]==t)&(played["winner"]==t)).sum()
            rows.append({
                "Team": t, "Played": len(played),
                "Wins": int(wins), "Losses": len(played)-int(wins),
                "Win %": round(wins/len(played)*100,1) if len(played)>0 else 0,
                "Toss Wins": int(toss_w),
                "Win After Toss %": round(toss_match_w/max(toss_w,1)*100,1)
            })
        return pd.DataFrame(rows).sort_values("Win %", ascending=False)

    tdf = team_stats_df(fmatches)

    # Active teams (enough matches)
    active = tdf[tdf["Played"] >= 20]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Wins", x=active["Team"], y=active["Wins"],
                             marker_color="#f0a500", hovertemplate="<b>%{x}</b><br>Wins: %{y}<extra></extra>"))
        fig.add_trace(go.Bar(name="Losses", x=active["Team"], y=active["Losses"],
                             marker_color="#e17055", hovertemplate="<b>%{x}</b><br>Losses: %{y}<extra></extra>"))
        fig.update_layout(barmode="group", xaxis_tickangle=-30)
        plot_layout(fig, "Wins vs Losses by Team", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Bar(
            x=active["Win %"], y=active["Team"],
            orientation="h",
            marker=dict(color=active["Win %"],
                        colorscale=[[0,"#e17055"],[0.5,"#ffeaa7"],[1,"#00b894"]]),
            text=active["Win %"].apply(lambda x: f"{x}%"),
            textposition="outside", textfont_color="#aabbcc",
            hovertemplate="<b>%{y}</b><br>Win Rate: %{x}%<extra></extra>"
        ))
        plot_layout(fig2, "Win Percentage by Team", height=400)
        fig2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Toss impact
        toss_bat = fmatches[fmatches["toss_decision"]=="bat"]
        toss_field = fmatches[fmatches["toss_decision"]=="field"]
        bat_won_pct = round((toss_bat["toss_winner"]==toss_bat["winner"]).sum()/max(len(toss_bat),1)*100,1)
        field_won_pct = round((toss_field["toss_winner"]==toss_field["winner"]).sum()/max(len(toss_field),1)*100,1)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=["Chose to Bat", "Chose to Field"],
            y=[bat_won_pct, field_won_pct],
            marker_color=["#45b7d1", "#f0a500"],
            text=[f"{bat_won_pct}%", f"{field_won_pct}%"],
            textposition="outside", textfont_color="#aabbcc",
            hovertemplate="<b>%{x}</b><br>Win %: %{y}%<extra></extra>"
        ))
        plot_layout(fig3, "Toss Decision Impact (Win % after Toss Win)", height=360)
        fig3.update_yaxes(range=[0, 80])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Season-wise wins for top teams
        top4 = tdf.head(4)["Team"].tolist()
        season_wins = []
        for _, row in fmatches.iterrows():
            if row["winner"] in top4:
                season_wins.append({"season": row["season"], "team": row["winner"]})
        sw_df = pd.DataFrame(season_wins).groupby(["season","team"]).size().reset_index(name="wins")
        fig4 = px.line(sw_df, x="season", y="wins", color="team",
                       color_discrete_sequence=COLORS, markers=True)
        plot_layout(fig4, "Season-wise Wins (Top 4 Teams)", height=360)
        fig4.update_xaxes(tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### 📊 Full Team Standings")
    tdf.index = range(1, len(tdf)+1)
    st.dataframe(tdf, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 — VENUES & MATCHES                                    ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[3]:
    st.markdown('<p class="section-header">VENUES & MATCH ANALYSIS</p>', unsafe_allow_html=True)

    # KPIs
    fi_scores = fdel[fdel["inning"]==1].groupby("match_id")["total_runs"].sum()
    avg_fi = fi_scores.mean()
    high_score = fi_scores.max()

    inn1 = fdel[fdel["inning"]==1].groupby("match_id")["batting_team"].first().reset_index()
    inn1.columns = ["match_id","batting_first"]
    mr = fmatches.merge(inn1, left_on="id", right_on="match_id", how="left")
    defend_w = (mr["winner"]==mr["batting_first"]).sum()
    chase_w = (mr["winner"]!=mr["batting_first"]).sum() - (mr["winner"].isna()).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Avg 1st Inn. Score", f"{avg_fi:.0f}"), unsafe_allow_html=True)
    k2.markdown(kpi("Highest Team Score", f"{int(high_score)}"), unsafe_allow_html=True)
    k3.markdown(kpi("Defended", f"{int(defend_w)}", "batting-first wins"), unsafe_allow_html=True)
    k4.markdown(kpi("Chased", f"{int(chase_w)}", "batting-second wins"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        venue_m = fmatches.groupby("venue").size().reset_index(name="matches")
        venue_m = venue_m.sort_values("matches", ascending=False).head(12)
        fig = go.Figure(go.Bar(
            x=venue_m["matches"], y=venue_m["venue"].str[:35],
            orientation="h",
            marker=dict(color=venue_m["matches"],
                        colorscale=[[0,"#6c5ce7"],[1,"#fd79a8"]]),
            text=venue_m["matches"],
            textposition="outside", textfont_color="#aabbcc",
            hovertemplate="<b>%{y}</b><br>Matches: %{x}<extra></extra>"
        ))
        plot_layout(fig, "Most Used Venues", height=420)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Chase vs Defend doughnut
        fig2 = go.Figure(go.Pie(
            labels=["Batting First (Defended)", "Batting Second (Chased)"],
            values=[int(defend_w), int(chase_w)],
            hole=0.5,
            marker=dict(colors=["#f0a500", "#4ecdc4"]),
            textfont=dict(size=13)
        ))
        plot_layout(fig2, "Chasing vs Defending Wins", height=420)
        fig2.update_layout(
            annotations=[dict(text=f"{int(defend_w)+int(chase_w)}<br><span style='font-size:12px'>Matches</span>",
                              x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#f0a500")]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Season score trend
    season_scores = fdel.merge(fmatches[["id","season"]], left_on="match_id", right_on="id", how="left")
    s_trend = season_scores[season_scores["inning"]==1].groupby(["match_id","season"])["total_runs"].sum().groupby("season").mean().round(1).reset_index()
    s_trend.columns = ["season","avg_score"]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=s_trend["season"], y=s_trend["avg_score"],
        mode="lines+markers+text",
        line=dict(color="#f0a500", width=3),
        marker=dict(size=10, color="#ff6b35", line=dict(color="#f0a500", width=2)),
        text=s_trend["avg_score"].round(1),
        textposition="top center", textfont=dict(color="#aabbcc", size=10),
        fill="tozeroy", fillcolor="rgba(240,165,0,0.08)",
        hovertemplate="Season: %{x}<br>Avg Score: %{y}<extra></extra>"
    ))
    plot_layout(fig3, "First Innings Average Score Trend by Season", height=340)
    fig3.update_xaxes(tickangle=-30)
    st.plotly_chart(fig3, use_container_width=True)

    # City analysis
    col3, col4 = st.columns(2)
    with col3:
        city_m = fmatches.dropna(subset=["city"]).groupby("city").size().reset_index(name="matches")
        city_m = city_m.sort_values("matches", ascending=False).head(10)
        fig4 = px.bar(city_m, x="city", y="matches", color="matches",
                      color_continuous_scale=[[0,"#6c5ce7"],[1,"#45b7d1"]],
                      text="matches")
        fig4.update_traces(textposition="outside", textfont_color="#aabbcc")
        plot_layout(fig4, "Matches by City", height=350)
        fig4.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        # Score heatmap by over
        over_runs = fdel[fdel["inning"]==1].groupby("over")["total_runs"].mean().reset_index()
        over_runs.columns = ["over","avg_runs"]
        over_runs["over_label"] = over_runs["over"] + 1
        fig5 = go.Figure(go.Bar(
            x=over_runs["over_label"], y=over_runs["avg_runs"],
            marker=dict(color=over_runs["avg_runs"],
                        colorscale=[[0,"#45b7d1"],[0.6,"#f0a500"],[1,"#e17055"]]),
            hovertemplate="Over %{x}<br>Avg runs: %{y:.1f}<extra></extra>"
        ))
        plot_layout(fig5, "Average Runs per Over (1st Innings)", height=350)
        st.plotly_chart(fig5, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 5 — WIN PREDICTOR                                       ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[4]:
    st.markdown('<p class="section-header">🤖 MATCH WIN PREDICTOR</p>', unsafe_allow_html=True)
    st.markdown("<p style='color:#8899bb'>ML-powered prediction using Random Forest, Logistic Regression & Decision Tree</p>", unsafe_allow_html=True)

    @st.cache_resource
    def train_models(m, d):
        """Train win prediction models using match-level features."""
        # Build feature set from deliveries (live match scenario)
        # Use first innings partial data to predict match outcome
        inn1 = d[d["inning"] == 1].copy()
        inn1_summary = inn1.groupby("match_id").agg(
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
            runs_scored=("total_runs", "sum"),
            wickets_fallen=("is_wicket", "sum"),
            balls_bowled=("ball", "count")
        ).reset_index()

        merged = inn1_summary.merge(
            m[["id","toss_winner","toss_decision","winner","venue"]],
            left_on="match_id", right_on="id"
        ).dropna(subset=["winner"])

        merged["toss_bat_first"] = (merged["toss_decision"] == "bat").astype(int)
        merged["batting_won"] = (merged["winner"] == merged["batting_team"]).astype(int)
        merged["overs_bowled"] = merged["balls_bowled"] // 6
        merged["run_rate"] = merged["runs_scored"] / merged["overs_bowled"].replace(0, 1)
        merged["wickets_left"] = 10 - merged["wickets_fallen"]

        # Encode teams
        le_bat = LabelEncoder()
        le_bowl = LabelEncoder()
        le_venue = LabelEncoder()

        all_teams = sorted(set(d["batting_team"].dropna().unique().tolist()))
        all_venues = sorted(m["venue"].dropna().unique().tolist())
        le_bat.fit(all_teams)
        le_bowl.fit(all_teams)
        le_venue.fit(all_venues)

        merged["bat_enc"] = le_bat.transform(merged["batting_team"].str.strip())
        merged["bowl_enc"] = le_bowl.transform(merged["bowling_team"].str.strip())
        merged["venue_enc"] = le_venue.transform(merged["venue"].fillna("Unknown"))

        features = ["bat_enc","bowl_enc","venue_enc","runs_scored",
                    "wickets_left","overs_bowled","run_rate","toss_bat_first"]
        X = merged[features]
        y = merged["batting_won"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        }
        trained = {}
        accs = {}
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            accs[name] = round(accuracy_score(y_test, clf.predict(X_test)) * 100, 1)
            trained[name] = clf

        return trained, accs, le_bat, le_bowl, le_venue, all_teams, all_venues, features

    with st.spinner("Training ML models..."):
        models, accs, le_bat, le_bowl, le_venue, all_teams, all_venues, features = train_models(fmatches, fdel)

    # Model accuracy display
    st.markdown("#### Model Accuracy")
    ma1, ma2, ma3 = st.columns(3)
    ma1.markdown(kpi("🌲 Random Forest", f"{accs['Random Forest']}%", "accuracy"), unsafe_allow_html=True)
    ma2.markdown(kpi("📈 Logistic Reg.", f"{accs['Logistic Regression']}%", "accuracy"), unsafe_allow_html=True)
    ma3.markdown(kpi("🌿 Decision Tree", f"{accs['Decision Tree']}%", "accuracy"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔮 Predict Match Outcome")
    st.markdown("<p style='color:#8899bb'>Enter live match situation to get win probability predictions</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        p1, p2 = st.columns(2)
        with p1:
            batting_team = st.selectbox("🏏 Batting Team", all_teams)
            venue = st.selectbox("🏟️ Venue", all_venues[:30])
            runs_scored = st.slider("Runs Scored", 0, 300, 80)
            wickets_fallen = st.slider("Wickets Fallen", 0, 10, 2)

        with p2:
            bowling_team_options = [t for t in all_teams if t != batting_team]
            bowling_team = st.selectbox("🎳 Bowling Team", bowling_team_options)
            overs_bowled = st.slider("Overs Bowled", 1, 20, 10)
            toss_bat_first = st.radio("Toss Winner Chose To", ["Bat", "Field"]) == "Bat"
            algorithm = st.selectbox("🤖 Algorithm", list(models.keys()))

        predict_btn = st.form_submit_button("🔮 Predict Winner", use_container_width=True)

    if predict_btn:
        run_rate = runs_scored / overs_bowled if overs_bowled > 0 else 0
        wickets_left = 10 - wickets_fallen

        # Encode safely
        bat_enc = le_bat.transform([batting_team])[0] if batting_team in le_bat.classes_ else 0
        bowl_enc = le_bowl.transform([bowling_team])[0] if bowling_team in le_bowl.classes_ else 0
        venue_enc = le_venue.transform([venue])[0] if venue in le_venue.classes_ else 0

        X_pred = pd.DataFrame([[bat_enc, bowl_enc, venue_enc, runs_scored,
                                 wickets_left, overs_bowled, run_rate, int(toss_bat_first)]],
                              columns=features)

        model = models[algorithm]
        prob = model.predict_proba(X_pred)[0]
        bat_win_prob = round(prob[1] * 100, 1)
        bowl_win_prob = round(prob[0] * 100, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown(f"""
            <div class="pred-box">
              <div style='color:#8899bb; font-size:12px; letter-spacing:1.5px; margin-bottom:12px;'>
                🏏 BATTING TEAM WIN PROBABILITY
              </div>
              <div style='font-family:Rajdhani; font-size:38px; font-weight:700;
                          color:{"#00b894" if bat_win_prob >= 50 else "#e17055"};'>
                {bat_win_prob}%
              </div>
              <div style='font-size:18px; color:#ccd6f6; margin-bottom:12px;'>{batting_team}</div>
              <div style='background:#1a2035; border-radius:6px; height:20px; overflow:hidden;'>
                <div style='background:{"linear-gradient(90deg,#00b894,#55efc4)" if bat_win_prob>=50 else "linear-gradient(90deg,#e17055,#fdcb6e)"};
                            height:100%; width:{bat_win_prob}%; border-radius:6px;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class="pred-box">
              <div style='color:#8899bb; font-size:12px; letter-spacing:1.5px; margin-bottom:12px;'>
                🎳 BOWLING TEAM WIN PROBABILITY
              </div>
              <div style='font-family:Rajdhani; font-size:38px; font-weight:700;
                          color:{"#00b894" if bowl_win_prob >= 50 else "#e17055"};'>
                {bowl_win_prob}%
              </div>
              <div style='font-size:18px; color:#ccd6f6; margin-bottom:12px;'>{bowling_team}</div>
              <div style='background:#1a2035; border-radius:6px; height:20px; overflow:hidden;'>
                <div style='background:{"linear-gradient(90deg,#00b894,#55efc4)" if bowl_win_prob>=50 else "linear-gradient(90deg,#e17055,#fdcb6e)"};
                            height:100%; width:{bowl_win_prob}%; border-radius:6px;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # All models comparison
        st.markdown("#### All Model Predictions")
        cols = st.columns(3)
        for i, (mname, clf) in enumerate(models.items()):
            p = clf.predict_proba(X_pred)[0]
            wp = round(p[1]*100, 1)
            lp = round(p[0]*100, 1)
            cols[i].markdown(f"""
            <div class="pred-box" style='text-align:center;'>
              <div style='color:#8899bb; font-size:11px; letter-spacing:1px;'>{mname}</div>
              <div style='font-family:Rajdhani; font-size:26px; font-weight:700; color:#f0a500; margin:8px 0;'>
                {batting_team if wp>=50 else bowling_team}
              </div>
              <div style='color:#{"00b894" if wp>=50 else "e17055"}; font-size:22px; font-weight:700;'>
                {max(wp,lp)}%
              </div>
              <div style='color:#8899bb; font-size:11px;'>Confidence</div>
            </div>
            """, unsafe_allow_html=True)

    # Feature importance (RF)
    rf = models["Random Forest"]
    fi_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_})
    fi_df = fi_df.sort_values("importance", ascending=True)
    feat_labels = {
        "bat_enc":"Batting Team","bowl_enc":"Bowling Team","venue_enc":"Venue",
        "runs_scored":"Runs Scored","wickets_left":"Wickets Left",
        "overs_bowled":"Overs Bowled","run_rate":"Run Rate","toss_bat_first":"Toss Decision"
    }
    fi_df["feature"] = fi_df["feature"].map(feat_labels)

    fig = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["feature"],
        orientation="h",
        marker=dict(color=fi_df["importance"],
                    colorscale=[[0,"#6c5ce7"],[1,"#f0a500"]]),
        text=fi_df["importance"].apply(lambda x: f"{x:.3f}"),
        textposition="outside", textfont_color="#aabbcc",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
    ))
    plot_layout(fig, "Random Forest — Feature Importance", height=360)
    st.plotly_chart(fig, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334466; font-size:12px; padding:10px 0'>
  IPL Analytics Dashboard · Data: 2008–2024 · Built with Streamlit + Plotly + Scikit-learn
</div>
""", unsafe_allow_html=True)
