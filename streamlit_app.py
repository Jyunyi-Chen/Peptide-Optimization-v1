import time
import config
import threading
import traceback

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

st.set_page_config(
    page_title="Peptide Optimizer",
    layout="wide", initial_sidebar_state="expanded",
)

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
ALL_MODELS = ["ACP", "AFP", "AMP", "AVP", "HEM"]
MODEL_DESCRIPTIONS = {
    "ACP": "Anticancer (maximise)",
    "AFP": "Antifungal (maximise)",
    "AMP": "Antimicrobial (maximise)",
    "AVP": "Antiviral (maximise)",
    "HEM": "Hemolysis (minimise)"
}

def _init_shared() -> dict:

    return \
    {   # idle | initializing | running | done | stopped | error
        "status": "idle", "episode": 0, "n_episodes": config.N_EPISODES,
        "results_df": None, "loss_data": None, "lr_data": None, "save_dir": None, "error": None,
    }

for _key, _default in [("shared", _init_shared()), ("stop_event", threading.Event()), ("training_thread", None)]:
    if _key not in st.session_state: st.session_state[_key] = _default

def _training_worker(shared: dict, stop_event: threading.Event) -> None:

    try:

        shared["status"] = "initializing"

        # Import here so config mutations applied in the main thread take effect.
        from peptide_optimization.framework import Framework

        framework = Framework()
        shared["save_dir"] = framework.save_dir
        shared["n_episodes"] = config.N_EPISODES
        shared["status"] = "running"

        def _on_episode_end(episode: int, df: pd.DataFrame, loss_data: dict, lr_data: list) -> None:

            shared["episode"] = episode
            shared["results_df"] = df
            shared["loss_data"] = loss_data
            shared["lr_data"] = lr_data

        framework.train(on_episode_end=_on_episode_end, stop_event=stop_event)

        shared["results_df"] = framework.exp_results_df.copy()
        shared["status"] = "stopped" if stop_event.is_set() else "done"

    except Exception:

        shared["status"] = "error"
        shared["error"] = traceback.format_exc()

def _validate_peptide(seq: str) -> str | None:

    if not seq: return "Sequence cannot be empty."

    invalid = sorted({aa for aa in seq if aa not in AMINO_ACIDS})

    return f"Invalid amino acid(s): {', '.join(invalid)}" if invalid else None

def _smooth(values: list[float], sigma: int) -> list[float]:

    return gaussian_filter1d(values, sigma=sigma).tolist() if len(values) > 3 else values

def _start_training(target: str, models: list[str], n_ep: int) -> None:

    thread = st.session_state.training_thread

    if thread and thread.is_alive():
        st.session_state.stop_event.set()
        thread.join(timeout=10)

    config.TARGET_PEPTIDE = target
    config.REWARD_MODELS = models
    config.N_EPISODES = n_ep

    st.session_state.shared = _init_shared()
    stop_event = threading.Event()
    st.session_state.stop_event = stop_event

    t = threading.Thread(
        target=_training_worker,
        args=(st.session_state.shared, stop_event),
        daemon=True,
    )
    t.start()
    st.session_state.training_thread = t

with st.sidebar:

    st.title("Peptide Optimizer")

    st.divider()
    st.title("Target Peptide Sequence")

    target_peptide = st.text_input(
        "Target Peptide Sequence",
        value=config.TARGET_PEPTIDE,
        label_visibility="collapsed",
    ).strip().upper()

    peptide_err = _validate_peptide(target_peptide)
    if peptide_err: st.error(peptide_err)

    st.divider()
    st.title("Reward Models")
    selected_models = []
    for m in ALL_MODELS:
        if st.checkbox(MODEL_DESCRIPTIONS[m], value=(m in config.REWARD_MODELS), key=f"chk_{m}"):
            selected_models.append(m)

    if not selected_models: st.error("Select at least one reward model.")

    status = st.session_state.shared["status"]
    is_active = status in ("initializing", "running")
    can_start = bool(target_peptide) and not peptide_err and len(selected_models) >= 1 and not is_active

    st.divider()
    if st.button("Start Training", disabled=not can_start, width="stretch", type="primary"):
        _start_training(target_peptide, selected_models, config.N_EPISODES)
        st.rerun()

    if is_active:
        if st.button("Stop Training", width="stretch"):
            st.session_state.stop_event.set()
            st.rerun()

    results_df: pd.DataFrame | None = st.session_state.shared.get("results_df")
    dl_disabled = results_df is None or len(results_df) == 0
    st.download_button(
        label="Download Training Logs (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8") if not dl_disabled else b"",
        file_name="training_logs.csv",
        mime="text/csv",
        disabled=dl_disabled,
        width="stretch",
    )

shared = st.session_state.shared
status = shared["status"]

STATUS_META = {
    "idle":         ("Idle",               "gray"),
    "initializing": ("Initializing...",    "orange"),
    "running":      ("Training",           "green"),
    "done":         ("Training Complete",  "blue"),
    "stopped":      ("Training Stopped",   "orange"),
    "error":        ("Error",              "red"),
}
label, color = STATUS_META.get(status, ("Unknown", "gray"))

hdr_col, dir_col = st.columns([4, 2])
with hdr_col:
    st.markdown(f"## :{color}[{label}]")
with dir_col:
    save_dir = shared.get("save_dir")

# Progress bar
episode = shared["episode"]
n_ep_total = shared["n_episodes"]
if status in ("running", "done", "stopped") and n_ep_total > 0:
    progress = min(episode / n_ep_total, 1.0)
    st.progress(progress, text=f"Episode {episode:,} / {n_ep_total:,}")
elif status == "initializing":
    st.progress(0.0, text="Loading models — this may take a moment...")
else:
    st.progress(0.0, text="Not started")

if status == "error" and shared.get("error"):
    st.error("Training failed with an exception:")
    st.code(shared["error"])

results_df = shared.get("results_df")
loss_data = shared.get("loss_data")
lr_data = shared.get("lr_data")

if results_df is not None and len(results_df) > 0:

    df = results_df
    active_prob_cols = [c for c in df.columns if c.endswith("-Prob_T")]
    active_models = [c.replace("-Prob_T", "") for c in active_prob_cols]

    # Metrics row
    metric_cols = st.columns(2 + len(active_models))
    with metric_cols[0]:
        st.metric("Episodes completed", f"{episode:,}")
    with metric_cols[1]:
        st.metric("Latest reward", f"{float(df['Cumulative-Reward'].iloc[-1]):+.4f}")
    for i, m in enumerate(active_models):
        with metric_cols[2 + i]:
            st.metric(f"{m} prob", f"{float(df[active_prob_cols[i]].iloc[-1]):.3f}")

    st.divider()

    sigma = 200

    tab_reward, tab_probs, tab_heuristic, tab_loss = st.tabs(
        ["Cumulative Reward", "Model Probabilities", "Heuristic Score", "Loss Curves"]
    )

    with tab_reward:
        y = _smooth(df["Cumulative-Reward"].astype(float).tolist(), sigma)
        fig = go.Figure(go.Scatter(y=y, mode="lines", line=dict(color="#1f77b4")))
        fig.update_layout(
            xaxis_title="Episode (index)", yaxis_title="Cumulative Reward",
            height=420, margin=dict(l=60, r=20, t=30, b=60),
        )
        st.plotly_chart(fig, width="stretch")

    with tab_probs:
        fig = go.Figure()
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
        for i, (m, col) in enumerate(zip(active_models, active_prob_cols)):
            y = _smooth(df[col].astype(float).tolist(), sigma)
            dash = "dot" if m == "HEM" else "solid"
            fig.add_trace(go.Scatter(
                y=y, mode="lines", name=m,
                line=dict(color=colors[i % len(colors)], dash=dash),
            ))
        fig.update_layout(
            xaxis_title="Episode (index)", yaxis_title="Probability",
            yaxis_range=[0, 1], height=420, margin=dict(l=60, r=20, t=30, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.caption("HEM (dotted) is minimised; all others are maximised.")
        st.plotly_chart(fig, width="stretch")

    with tab_heuristic:
        y = _smooth(df["Heuristic_T"].astype(float).tolist(), sigma)
        fig = go.Figure(go.Scatter(y=y, mode="lines", line=dict(color="#2ca02c")))
        fig.update_layout(
            xaxis_title="Episode (index)", yaxis_title="Heuristic Score",
            height=420, margin=dict(l=60, r=20, t=30, b=60),
        )
        st.plotly_chart(fig, width="stretch")

    with tab_loss:
        if loss_data and loss_data.get("actor1_loss"):
            loss_sigma = 1
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    "Actor1 Loss", "Actor2 Loss",
                    "Critic Loss", "Entropy Bonus 1",
                    "Entropy Bonus 2", "Learning Rate",
                ],
            )
            panels = [
                ("actor1_loss", 1, 1), ("actor2_loss", 1, 2),
                ("critic_loss", 2, 1), ("entropy1", 2, 2),
                ("entropy2", 3, 1),
            ]
            for key, row, col in panels:
                y = _smooth(loss_data[key], loss_sigma)
                fig.add_trace(go.Scatter(y=y, mode="lines", showlegend=False), row=row, col=col)
            if lr_data:
                y = _smooth(lr_data, loss_sigma)
                fig.add_trace(go.Scatter(y=y, mode="lines", showlegend=False), row=3, col=2)
            fig.update_layout(height=700, margin=dict(l=50, r=20, t=50, b=40))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Loss data will appear after the first learning step.")

else:
    if status == "idle":
        st.info("Configure the settings in the sidebar and click **Start Training** to begin.")
    elif status == "initializing":
        st.info("Loading models — charts will appear once training starts.")

if status in ("initializing", "running"):
    time.sleep(2)
    st.rerun()
