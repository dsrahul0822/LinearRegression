import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

DARK_BG  = "#0f172a"
PANEL_BG = "#1e293b"
GRID_COL = "#334155"
TEXT_COL = "#94a3b8"

FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
TARGET   = "Profit"


def _mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def render():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🧪 Model <span class="accent">Training & Evaluation</span></h1>
            <p>Configure the train/test split, fit the regression model, and review performance metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "df" not in st.session_state:
        st.markdown(
            "<div class='warning-box'>⚠️ No dataset found. Please go to "
            "<strong>📂 Upload Data</strong> first.</div>",
            unsafe_allow_html=True,
        )
        return

    df: pd.DataFrame = st.session_state["df"]
    n_total = len(df)

    # ── Configuration panel ───────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'><h2>Configuration</h2>"
        "<p>Adjust training parameters</p></div>",
        unsafe_allow_html=True,
    )

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        test_pct = st.slider(
            "Test set size (%)",
            min_value=10, max_value=40, value=20, step=5,
            help="Percentage of rows reserved for testing",
        )
    with cfg2:
        rand_seed = st.number_input(
            "Random seed",
            min_value=0, max_value=999, value=42, step=1,
            help="Seed for reproducible splits",
        )
    with cfg3:
        n_train = int(n_total * (1 - test_pct / 100))
        n_test  = n_total - n_train
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:1.6rem;">
                <div class="metric-label">Split preview</div>
                <div class="metric-value" style="font-size:1.1rem;">
                    {n_train} train / {n_test} test
                </div>
                <div class="metric-sub">{100 - test_pct}% / {test_pct}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    train_btn = st.button("🚀  Train Model", use_container_width=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    if train_btn:
        X = df[FEATURES].values
        y = df[TARGET].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct / 100, random_state=int(rand_seed)
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = _mape(y_test, y_pred)

        # Persist
        st.session_state["model"]       = model
        st.session_state["X_test"]      = X_test
        st.session_state["y_test"]      = y_test
        st.session_state["y_pred"]      = y_pred
        st.session_state["split_ratio"] = test_pct
        st.session_state["metrics"]     = dict(R2=r2, MAE=mae, MAPE=mape, MSE=mse, RMSE=rmse)
        st.session_state["coef"]        = dict(zip(FEATURES, model.coef_))
        st.session_state["intercept"]   = model.intercept_

    # ── Results (shown if model exists) ───────────────────────────────────────
    if "metrics" not in st.session_state:
        st.markdown(
            "<div class='info-box'>ℹ️ Configure the split above and click <strong>Train Model</strong> to see results.</div>",
            unsafe_allow_html=True,
        )
        return

    metrics  = st.session_state["metrics"]
    y_test   = st.session_state["y_test"]
    y_pred   = st.session_state["y_pred"]
    model    = st.session_state["model"]

    st.markdown(
        "<div class='success-box'>✅ Model trained successfully on the training set and evaluated on the held-out test set.</div>",
        unsafe_allow_html=True,
    )

    # ── Metric cards ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'><h2>Performance Metrics</h2>"
        "<p>Evaluated on the test set only</p></div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    metric_defs = [
        (m1, "R²",   f"{metrics['R2']:.4f}",      "Coefficient of Determination",  "Higher is better (max 1.0)"),
        (m2, "MAE",  f"${metrics['MAE']:,.0f}",    "Mean Absolute Error",            "Avg absolute deviation"),
        (m3, "MAPE", f"{metrics['MAPE']:.2f}%",    "Mean Abs. % Error",              "Lower is better"),
        (m4, "MSE",  f"${metrics['MSE']:,.0f}",    "Mean Squared Error",             "Penalises large errors"),
        (m5, "RMSE", f"${metrics['RMSE']:,.0f}",   "Root Mean Squared Error",        "Same unit as target"),
    ]
    for col, label, value, full, sub in metric_defs:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{full}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    tab_actual, tab_resid, tab_coef = st.tabs(
        ["  📈  Actual vs Predicted  ", "  📉  Residuals  ", "  🔢  Coefficients  "]
    )

    with tab_actual:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _style_ax(fig, ax)

        ax.scatter(y_test, y_pred, color="#38bdf8", alpha=0.8, s=60, label="Predictions", zorder=3)
        lo = min(y_test.min(), y_pred.min()) * 0.95
        hi = max(y_test.max(), y_pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color="#f472b6", linewidth=1.5, linestyle="--", label="Perfect fit")
        ax.set_xlabel("Actual Profit", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Predicted Profit", color=TEXT_COL, fontsize=9)
        ax.set_title("Actual vs Predicted Profit", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab_resid:
        residuals = y_test - y_pred
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig2.patch.set_facecolor(DARK_BG)
        for ax in [ax1, ax2]:
            _style_ax(fig2, ax)

        ax1.scatter(y_pred, residuals, color="#818cf8", alpha=0.8, s=55, zorder=3)
        ax1.axhline(0, color="#f472b6", linewidth=1.5, linestyle="--")
        ax1.set_xlabel("Predicted Profit", color=TEXT_COL, fontsize=9)
        ax1.set_ylabel("Residuals", color=TEXT_COL, fontsize=9)
        ax1.set_title("Residuals vs Fitted", color="#f8fafc", fontsize=10, fontweight="bold")
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))

        ax2.hist(residuals, bins=10, color="#34d399", alpha=0.75, edgecolor=PANEL_BG)
        ax2.axvline(0, color="#f472b6", linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Residual Value", color=TEXT_COL, fontsize=9)
        ax2.set_ylabel("Frequency", color=TEXT_COL, fontsize=9)
        ax2.set_title("Residual Distribution", color="#f8fafc", fontsize=10, fontweight="bold")
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))

        plt.tight_layout(pad=1.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with tab_coef:
        coef = st.session_state["coef"]
        intercept = st.session_state["intercept"]

        fig3, ax = plt.subplots(figsize=(6, 3))
        _style_ax(fig3, ax)

        names  = list(coef.keys())
        values = list(coef.values())
        colors = ["#22c55e" if v > 0 else "#ef4444" for v in values]
        bars   = ax.barh(names, values, color=colors, alpha=0.85, edgecolor=PANEL_BG)
        ax.axvline(0, color=TEXT_COL, linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(
                val + (max(values) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", color="#f1f5f9", fontsize=8,
            )
        ax.set_xlabel("Coefficient value", color=TEXT_COL, fontsize=9)
        ax.set_title("Feature Coefficients", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.markdown(
            f"<div class='info-box' style='margin-top:0.8rem;'>📌 <strong>Intercept:</strong> "
            f"${intercept:,.2f}</div>",
            unsafe_allow_html=True,
        )

    # ── Detailed predictions table ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'><h2>Test-Set Predictions</h2>"
        "<p>Row-level comparison of actual vs predicted values</p></div>",
        unsafe_allow_html=True,
    )
    results_df = pd.DataFrame({
        "Actual Profit ($)":    y_test,
        "Predicted Profit ($)": y_pred,
        "Error ($)":            y_test - y_pred,
        "Abs Error ($)":        np.abs(y_test - y_pred),
        "% Error":              np.abs((y_test - y_pred) / y_test) * 100,
    })
    st.dataframe(
        results_df.style
            .format({
                "Actual Profit ($)":    "${:,.2f}",
                "Predicted Profit ($)": "${:,.2f}",
                "Error ($)":            "${:,.2f}",
                "Abs Error ($)":        "${:,.2f}",
                "% Error":              "{:.2f}%",
            })
            .background_gradient(subset=["Abs Error ($)"], cmap="Reds"),
        use_container_width=True,
    )

    st.markdown(
        "<div class='info-box' style='margin-top:1rem;'>👉 Head to "
        "<strong>🔮 Predict Profit</strong> to make custom predictions.</div>",
        unsafe_allow_html=True,
    )


def _style_ax(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
