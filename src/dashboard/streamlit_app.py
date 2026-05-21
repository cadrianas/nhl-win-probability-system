"""
Phase 7: NHL Win Probability Dashboard
=======================================

Interactive Streamlit app that visualises calibrated win probability over
the course of a game, with xG momentum, goal markers, and live game state.

Run from project root:
    streamlit run src/dashboard/streamlit_app.py

Requirements (add to requirements.txt if missing):
    streamlit>=1.32
    plotly>=5.18
    shap>=0.44
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — same pattern as every other script in the project
# ---------------------------------------------------------------------------

SRC_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(SRC_PATH))

from utils.paths import (
    DATA_INTERIM,
    DATA_PROCESSED,
    RESULTS_MODELS,
    ensure_directories,
)

ensure_directories()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NHL Win Probability",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIOD_DURATION_S = 1200   # 20 minutes per period
OT_DURATION_S     = 300    # 5-minute OT

# Map numeric period → game-clock seconds elapsed at start of that period
PERIOD_OFFSET = {1: 0, 2: 1200, 3: 2400, 4: 3600}

HOME_COLOR = "#1f77b4"   # blue
AWAY_COLOR = "#d62728"   # red


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = RESULTS_MODELS / "xgboost_calibrated_isotonic.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading game states…")
def load_game_states() -> pd.DataFrame:
    path = DATA_INTERIM / "game_states.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df


@st.cache_data(show_spinner="Loading shots…")
def load_shots() -> pd.DataFrame:
    path = DATA_INTERIM / "shots.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df


# ---------------------------------------------------------------------------
# Helper: compute time_elapsed_s if absent
# ---------------------------------------------------------------------------

def add_time_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a time_elapsed_s column (seconds into game)."""
    if "time_elapsed_s" in df.columns:
        return df
    if "time_elapsed" in df.columns:
        # Assume column is already in seconds; rename for consistency
        df = df.rename(columns={"time_elapsed": "time_elapsed_s"})
        return df
    # Build from period + time_remaining_seconds
    if "period" in df.columns and "time_remaining_seconds" in df.columns:
        df["time_elapsed_s"] = (
            df["period"].map(PERIOD_OFFSET).fillna(3600)
            + (PERIOD_DURATION_S - df["time_remaining_seconds"]).clip(lower=0)
        )
    return df


# ---------------------------------------------------------------------------
# Helper: run model on a game's rows
# ---------------------------------------------------------------------------

def predict_game(model, game_df: pd.DataFrame) -> np.ndarray:
    """
    Run the calibrated model on the numeric features of a game's rows.
    Returns array of home-win probabilities, one per row.
    """
    X = game_df.select_dtypes(include=[np.number]).drop(
        columns=["target_home_win", "time_elapsed_s"], errors="ignore"
    )
    if X.empty or model is None:
        return np.full(len(game_df), 0.5)
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return np.full(len(game_df), 0.5)


# ---------------------------------------------------------------------------
# Helper: team label
# ---------------------------------------------------------------------------

def team_label(df: pd.DataFrame, side: str) -> str:
    col = f"{side}_team_code"
    if col in df.columns:
        return str(df[col].iloc[0])
    return side.title()


# ---------------------------------------------------------------------------
# Plotly: win probability chart
# ---------------------------------------------------------------------------

def win_prob_chart(
    game_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    goals_df: pd.DataFrame,
) -> go.Figure:
    t = game_df["time_elapsed_s"] / 60   # minutes

    fig = go.Figure()

    # Home probability band fill
    fig.add_trace(go.Scatter(
        x=t, y=game_df["home_win_prob"],
        fill="tozeroy",
        fillcolor=f"rgba(31, 119, 180, 0.12)",
        line=dict(color=HOME_COLOR, width=2.5),
        name=home_team,
        hovertemplate="%{y:.1%}<extra>" + home_team + "</extra>",
    ))

    # Away probability (just line, no fill — would overlap)
    fig.add_trace(go.Scatter(
        x=t, y=1 - game_df["home_win_prob"],
        line=dict(color=AWAY_COLOR, width=2.5),
        name=away_team,
        hovertemplate="%{y:.1%}<extra>" + away_team + "</extra>",
    ))

    # 50-50 line
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="gray", line_width=1,
        annotation_text="50 / 50", annotation_position="right",
    )

    # Period separators
    for p_end_min in [20, 40]:
        fig.add_vline(
            x=p_end_min, line_dash="dash", line_color="lightgray", line_width=1,
        )

    # Goal markers
    if not goals_df.empty and "time_elapsed_s" in goals_df.columns:
        for _, goal in goals_df.iterrows():
            gtime = goal["time_elapsed_s"] / 60
            is_home = str(goal.get("team", "")).upper() == home_team.upper()
            color = HOME_COLOR if is_home else AWAY_COLOR
            scorer = str(goal.get("shooterName", goal.get("team", "Goal")))
            fig.add_vline(
                x=gtime,
                line_color=color,
                line_width=2,
                annotation_text=f"⚽ {scorer}" if len(scorer) < 20 else "⚽ GOAL",
                annotation_textangle=-90,
                annotation_font_size=10,
                annotation_font_color=color,
            )

    fig.update_layout(
        title=dict(text=f"Win Probability — {away_team} @ {home_team}", font_size=16),
        xaxis=dict(title="Minutes elapsed", range=[0, t.max() + 1], ticksuffix="'"),
        yaxis=dict(title="Win probability", tickformat=".0%", range=[0, 1]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        margin=dict(t=60, b=40, l=60, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: xG differential chart
# ---------------------------------------------------------------------------

def xg_diff_chart(
    game_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> go.Figure:
    if "home_xg" not in game_df.columns or "away_xg" not in game_df.columns:
        return None

    t    = game_df["time_elapsed_s"] / 60
    diff = game_df["home_xg"] - game_df["away_xg"]

    pos  = diff.clip(lower=0)
    neg  = diff.clip(upper=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=pos, fill="tozeroy",
        fillcolor=f"rgba(31, 119, 180, 0.25)",
        line=dict(color=HOME_COLOR, width=1),
        name=f"{home_team} ahead",
        hovertemplate="xG diff: %{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=neg, fill="tozeroy",
        fillcolor=f"rgba(214, 39, 40, 0.25)",
        line=dict(color=AWAY_COLOR, width=1),
        name=f"{away_team} ahead",
        hovertemplate="xG diff: %{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="gray", line_width=1)

    for p_end_min in [20, 40]:
        fig.add_vline(x=p_end_min, line_dash="dash", line_color="lightgray", line_width=1)

    fig.update_layout(
        title=dict(text=f"Cumulative xG Differential ({home_team} − {away_team})", font_size=14),
        xaxis=dict(title="Minutes elapsed", ticksuffix="'"),
        yaxis=dict(title="xG differential"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=280,
        margin=dict(t=50, b=40, l=60, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: shot momentum chart (rolling shots last 5 min)
# ---------------------------------------------------------------------------

def momentum_chart(game_df: pd.DataFrame, home_team: str, away_team: str):
    h_col = next((c for c in ["shots_last_5min_home", "shots_5min_home"] if c in game_df.columns), None)
    a_col = next((c for c in ["shots_last_5min_away", "shots_5min_away"] if c in game_df.columns), None)
    if h_col is None or a_col is None:
        return None

    t = game_df["time_elapsed_s"] / 60

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=game_df[h_col],
        line=dict(color=HOME_COLOR, width=2),
        name=f"{home_team} shots (5 min)",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=game_df[a_col],
        line=dict(color=AWAY_COLOR, width=2),
        name=f"{away_team} shots (5 min)",
    ))
    fig.update_layout(
        title=dict(text="Shot Momentum (rolling 5-minute window)", font_size=14),
        xaxis=dict(title="Minutes elapsed", ticksuffix="'"),
        yaxis=dict(title="Shots (last 5 min)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=250,
        margin=dict(t=50, b=40, l=60, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Scorecard metric row
# ---------------------------------------------------------------------------

def scorecard(game_df: pd.DataFrame, home_team: str, away_team: str) -> None:
    last = game_df.iloc[-1]

    # Derive score from score_differential if raw scores absent
    score_diff = int(last.get("score_differential", 0))
    home_goals = int(last.get("homeTeamGoals", last.get("home_goals", 0)))
    away_goals = int(last.get("awayTeamGoals", last.get("away_goals", home_goals - score_diff)))

    home_xg = float(last.get("home_xg", 0))
    away_xg = float(last.get("away_xg", 0))

    home_prob = float(last.get("home_win_prob", 0.5))

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(f"🏒 {home_team}", f"{home_goals} G", f"xG {home_xg:.2f}")
    col2.metric(f"🏒 {away_team}", f"{away_goals} G", f"xG {away_xg:.2f}")
    col3.metric("Home win prob", f"{home_prob:.1%}")
    col4.metric("Away win prob", f"{1 - home_prob:.1%}")

    outcome = last.get("target_home_win", None)
    if outcome is not None:
        result = f"{'🏠 ' + home_team} won" if int(outcome) == 1 else f"{'✈️ ' + away_team} won"
        col5.metric("Final result", result)


# ---------------------------------------------------------------------------
# SHAP waterfall for a single snapshot
# ---------------------------------------------------------------------------

def shap_panel(model, game_df: pd.DataFrame, row_idx: int) -> None:
    try:
        import shap
    except ImportError:
        st.info("Install `shap` to enable explainability: `pip install shap`")
        return

    X = game_df.select_dtypes(include=[np.number]).drop(
        columns=["target_home_win", "time_elapsed_s"], errors="ignore"
    )
    if X.empty:
        return

    row = X.iloc[[row_idx]]

    # TreeExplainer works directly on the base XGBoost inside the calibrated wrapper
    base_model = getattr(model, "estimator", model)
    try:
        explainer = shap.TreeExplainer(base_model)
        sv = explainer.shap_values(row)
        shap_vals = sv[0] if isinstance(sv, list) else sv[0]
    except Exception as e:
        st.warning(f"SHAP unavailable for this model: {e}")
        return

    feat_names = X.columns.tolist()
    df_shap = (
        pd.DataFrame({"feature": feat_names, "shap": shap_vals})
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(10)
        .sort_values("shap")
    )

    colors = [HOME_COLOR if v > 0 else AWAY_COLOR for v in df_shap["shap"]]

    fig = go.Figure(go.Bar(
        x=df_shap["shap"],
        y=df_shap["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top feature contributions at this moment", font_size=13),
        xaxis_title="SHAP value (→ favours home, ← favours away)",
        height=320,
        margin=dict(t=50, b=40, l=180, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🏒 NHL Win Probability Dashboard")
    st.caption("Calibrated XGBoost model · Phases 1–5 of the NHL Win Probability Project")

    # ── Load assets ─────────────────────────────────────────────────────────
    model      = load_model()
    game_states = load_game_states()
    shots_all  = load_shots()

    if game_states.empty:
        st.error(
            "No game states found. Expected: `data/interim/game_states.csv`\n\n"
            "Run Phase 1 (game state generation) first."
        )
        return

    if model is None:
        st.warning(
            "Calibrated model not found at `results/models/xgboost_calibrated_isotonic.pkl`. "
            "Predictions will default to 50%."
        )

    game_states = add_time_elapsed(game_states)

    # ── Sidebar: game selector ───────────────────────────────────────────────
    st.sidebar.header("Game selector")

    # Season filter
    if "season" in game_states.columns:
        seasons = sorted(game_states["season"].dropna().unique(), reverse=True)
        selected_season = st.sidebar.selectbox("Season", seasons)
        season_df = game_states[game_states["season"] == selected_season]
    else:
        season_df = game_states

    # Playoff toggle
    if "is_playoff_game" in season_df.columns:
        show_playoffs = st.sidebar.toggle("Playoff games only", value=False)
        if show_playoffs:
            season_df = season_df[season_df["is_playoff_game"] == 1]
        else:
            season_df = season_df[season_df["is_playoff_game"] == 0]

    # Team filter
    team_cols = [c for c in ["home_team_code", "away_team_code"] if c in season_df.columns]
    if team_cols:
        all_teams = sorted(
            pd.concat([season_df[c] for c in team_cols]).dropna().unique()
        )
        selected_team = st.sidebar.selectbox("Filter by team (optional)", ["All"] + all_teams)
        if selected_team != "All":
            mask = (season_df.get("home_team_code", "") == selected_team) | \
                   (season_df.get("away_team_code", "") == selected_team)
            season_df = season_df[mask]

    # Game selector — build a readable label
    game_ids = season_df["game_id"].unique()

    if len(game_ids) == 0:
        st.sidebar.warning("No games match the current filters.")
        return

    def game_label(gid: int) -> str:
        g = season_df[season_df["game_id"] == gid].iloc[0]
        home = g.get("home_team_code", "HOME")
        away = g.get("away_team_code", "AWAY")
        date = g.get("game_date", g.get("date", ""))
        return f"{date}  {away} @ {home}" if date else f"{away} @ {home}  (ID {gid})"

    game_labels = {gid: game_label(gid) for gid in game_ids}
    selected_label = st.sidebar.selectbox(
        "Select game",
        options=list(game_labels.values()),
    )
    selected_game_id = next(gid for gid, lbl in game_labels.items() if lbl == selected_label)

    # ── Slice the selected game ──────────────────────────────────────────────
    game_df = (
        season_df[season_df["game_id"] == selected_game_id]
        .sort_values("time_elapsed_s")
        .reset_index(drop=True)
    )

    home_team = team_label(game_df, "home")
    away_team = team_label(game_df, "away")

    # Run model predictions
    game_df["home_win_prob"] = predict_game(model, game_df)

    # Goals for this game
    if not shots_all.empty and "game_id" in shots_all.columns:
        goals_df = shots_all[
            (shots_all["game_id"] == selected_game_id) &
            (shots_all.get("isGoal", shots_all.get("is_goal", pd.Series(0, index=shots_all.index))) == 1)
        ].copy()
        goals_df = add_time_elapsed(goals_df)
    else:
        goals_df = pd.DataFrame()

    # ── Scorecard row ────────────────────────────────────────────────────────
    scorecard(game_df, home_team, away_team)

    st.divider()

    # ── Win probability chart ────────────────────────────────────────────────
    st.plotly_chart(
        win_prob_chart(game_df, home_team, away_team, goals_df),
        use_container_width=True,
    )

    # ── xG + momentum charts side by side ───────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        xg_fig = xg_diff_chart(game_df, home_team, away_team)
        if xg_fig:
            st.plotly_chart(xg_fig, use_container_width=True)
        else:
            st.info("xG columns (`home_xg`, `away_xg`) not found in game states.")

    with col_right:
        mom_fig = momentum_chart(game_df, home_team, away_team)
        if mom_fig:
            st.plotly_chart(mom_fig, use_container_width=True)
        else:
            st.info("Shot momentum columns not found in game states.")

    st.divider()

    # ── SHAP explainability (expandable) ────────────────────────────────────
    with st.expander("🔍 Feature explainability — what's driving the prediction?"):
        st.caption(
            "Select a moment in the game to see which features pushed the model "
            "toward a home or away win prediction."
        )

        max_min = int(game_df["time_elapsed_s"].max() / 60)
        snap_minute = st.slider(
            "Game minute to explain", min_value=1, max_value=max(max_min, 1), value=max(max_min // 2, 1)
        )

        # Find nearest row to selected minute
        snap_seconds = snap_minute * 60
        row_idx = (game_df["time_elapsed_s"] - snap_seconds).abs().argmin()

        snap_row = game_df.iloc[row_idx]
        st.write(
            f"**Snapshot:** minute {snap_minute} · "
            f"Period {int(snap_row.get('period', '?'))} · "
            f"Score differential: {int(snap_row.get('score_differential', 0)):+d} · "
            f"Predicted home win probability: **{snap_row['home_win_prob']:.1%}**"
        )

        if model is not None:
            shap_panel(model, game_df, row_idx)

    # ── Raw game state table (expandable) ───────────────────────────────────
    with st.expander("📋 Raw game state data"):
        display_cols = [c for c in [
            "time_elapsed_s", "period", "time_remaining_seconds",
            "score_differential", "home_xg", "away_xg",
            "strength_state", "home_goalie_pulled", "away_goalie_pulled",
            "home_win_prob",
        ] if c in game_df.columns]
        st.dataframe(
            game_df[display_cols].rename(columns={"time_elapsed_s": "elapsed (s)"}),
            use_container_width=True,
            height=300,
        )

    # ── Sidebar: model info ──────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Model info")
    st.sidebar.markdown(
        "**XGBoost** · Isotonic calibration  \n"
        "Trained on 2014–2025 regular season  \n"
        f"Validation AUC: **0.7221**  \n"
        f"Snapshots this game: **{len(game_df):,}**"
    )

    if model is None:
        st.sidebar.error("Model not loaded — showing flat 50% probability")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()