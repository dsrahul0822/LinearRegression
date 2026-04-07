import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DARK_BG  = "#0f172a"
PANEL_BG = "#1e293b"
GRID_COL = "#334155"
TEXT_COL = "#94a3b8"

FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
TARGET   = "Profit"


def render():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🔮 Predict <span class="accent">Profit</span></h1>
            <p>Enter your startup's spending values below to get an instant profit estimate from the trained model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "model" not in st.session_state:
        st.markdown(
            "<div class='warning-box'>⚠️ No trained model found. Please complete "
            "<strong>🧪 Model Training</strong> first.</div>",
            unsafe_allow_html=True,
        )
        return

    model   = st.session_state["model"]
    df      = st.session_state.get("df")
    metrics = st.session_state.get("metrics", {})

    # ── Model summary banner ──────────────────────────────────────────────────
    if metrics:
        s1, s2, s3 = st.columns(3)
        quick = [
            (s1, "R²",   f"{metrics['R2']:.4f}",       "Model accuracy"),
            (s2, "RMSE", f"${metrics['RMSE']:,.0f}",    "Avg error magnitude"),
            (s3, "MAPE", f"{metrics['MAPE']:.2f}%",     "Avg % deviation"),
        ]
        for col, label, val, sub in quick:
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'><h2>Enter Spending Values</h2>"
        "<p>All values in US Dollars ($)</p></div>",
        unsafe_allow_html=True,
    )

    # Compute sensible defaults from dataset stats
    rd_def    = float(df["R&D Spend"].mean())       if df is not None else 100000.0
    admin_def = float(df["Administration"].mean())  if df is not None else 120000.0
    mkt_def   = float(df["Marketing Spend"].mean()) if df is not None else 200000.0

    rd_min    = 0.0
    rd_max    = float(df["R&D Spend"].max() * 1.5)       if df is not None else 500000.0
    admin_max = float(df["Administration"].max() * 1.5)  if df is not None else 300000.0
    mkt_max   = float(df["Marketing Spend"].max() * 1.5) if df is not None else 600000.0

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🔬 R&D Spend")
            rd_input = st.number_input(
                "Amount ($)",
                min_value=rd_min,
                max_value=rd_max,
                value=rd_def,
                step=1000.0,
                format="%.2f",
                key="rd_input",
                label_visibility="collapsed",
            )
            st.slider(
                "R&D slider",
                min_value=rd_min,
                max_value=rd_max,
                value=rd_def,
                step=1000.0,
                format="$%.0f",
                key="rd_slider",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("##### 🏢 Administration")
            admin_input = st.number_input(
                "Amount ($)",
                min_value=0.0,
                max_value=admin_max,
                value=admin_def,
                step=1000.0,
                format="%.2f",
                key="admin_input",
                label_visibility="collapsed",
            )
            st.slider(
                "Admin slider",
                min_value=0.0,
                max_value=admin_max,
                value=admin_def,
                step=1000.0,
                format="$%.0f",
                key="admin_slider",
                label_visibility="collapsed",
            )

        with col3:
            st.markdown("##### 📣 Marketing Spend")
            mkt_input = st.number_input(
                "Amount ($)",
                min_value=0.0,
                max_value=mkt_max,
                value=mkt_def,
                step=1000.0,
                format="%.2f",
                key="mkt_input",
                label_visibility="collapsed",
            )
            st.slider(
                "Marketing slider",
                min_value=0.0,
                max_value=mkt_max,
                value=mkt_def,
                step=1000.0,
                format="$%.0f",
                key="mkt_slider",
                label_visibility="collapsed",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Predict Profit", use_container_width=True)

    # ── Prediction output ─────────────────────────────────────────────────────
    if submitted:
        X_new = np.array([[rd_input, admin_input, mkt_input]])
        pred  = model.predict(X_new)[0]

        st.markdown(
            f"""
            <div class="prediction-result">
                <div class="pred-label">Estimated Net Profit</div>
                <div class="pred-value">${pred:,.2f}</div>
                <div class="pred-sub">
                    Based on: R&D ${rd_input:,.0f} · Admin ${admin_input:,.0f} · Marketing ${mkt_input:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence context using RMSE
        if metrics:
            rmse = metrics["RMSE"]
            lo   = pred - rmse
            hi   = pred + rmse
            st.markdown(
                f"""
                <div class="info-box" style="margin-top:1rem;">
                    📊 <strong>Confidence range (±1 RMSE):</strong>
                    ${lo:,.2f} – ${hi:,.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Visual context: where does this prediction sit? ───────────────────
        if df is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-header'><h2>Prediction in Context</h2>"
                "<p>How your prediction compares to the training data</p></div>",
                unsafe_allow_html=True,
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(PANEL_BG)

            ax.hist(
                df[TARGET],
                bins=15,
                color="#38bdf8",
                alpha=0.6,
                edgecolor=PANEL_BG,
                label="Dataset profit distribution",
            )
            ax.axvline(pred, color="#f472b6", linewidth=2.5, linestyle="--", label=f"Your prediction: ${pred:,.0f}")
            ax.axvline(df[TARGET].mean(), color="#fcd34d", linewidth=1.5, linestyle=":", label=f"Dataset mean: ${df[TARGET].mean():,.0f}")

            ax.set_xlabel("Profit ($)", color=TEXT_COL, fontsize=9)
            ax.set_ylabel("Count", color=TEXT_COL, fontsize=9)
            ax.set_title("Predicted Profit vs Dataset Distribution", color="#f8fafc", fontsize=11, fontweight="bold")
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COL)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Input breakdown ───────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-header'><h2>Spend Breakdown</h2>"
                "<p>Your inputs vs dataset averages</p></div>",
                unsafe_allow_html=True,
            )
            comp_cols = st.columns(3)
            inputs = {
                "R&D Spend":      (rd_input,    df["R&D Spend"].mean()),
                "Administration": (admin_input, df["Administration"].mean()),
                "Marketing Spend":(mkt_input,   df["Marketing Spend"].mean()),
            }
            icons = ["🔬", "🏢", "📣"]
            for (feat, (user_val, avg_val)), col, icon in zip(inputs.items(), comp_cols, icons):
                diff = user_val - avg_val
                diff_pct = (diff / avg_val) * 100 if avg_val else 0
                sign  = "▲" if diff >= 0 else "▼"
                color = "#22c55e" if diff >= 0 else "#ef4444"
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{icon} {feat}</div>
                            <div class="metric-value" style="font-size:1.2rem;">${user_val:,.0f}</div>
                            <div class="metric-sub">Dataset avg: ${avg_val:,.0f}</div>
                            <div style="color:{color}; font-size:0.85rem; font-weight:600; margin-top:4px;">
                                {sign} {abs(diff_pct):.1f}% vs avg
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
