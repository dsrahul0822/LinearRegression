import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from io import BytesIO


# ── Seaborn dark theme ────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="deep")
DARK_BG   = "#0f172a"
PANEL_BG  = "#1e293b"
GRID_COL  = "#334155"
TEXT_COL  = "#94a3b8"
ACCENT    = "#38bdf8"
PALETTE   = ["#38bdf8", "#818cf8", "#34d399", "#fb7185"]


def _apply_dark_style(fig, axes_flat):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_flat:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))


def render():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📊 Data <span class="accent">Analysis</span></h1>
            <p>Explore feature distributions and pairwise relationships across all variables.</p>
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
    features = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'><h2>Pair Plot Settings</h2>"
        "<p>Customise the visualisation</p></div>",
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        plot_kind = st.selectbox(
            "Diagonal plot type",
            ["KDE (density)", "Histogram"],
            help="What to show on the diagonal cells",
        )
    with ctrl2:
        scatter_alpha = st.slider("Scatter opacity", 0.3, 1.0, 0.7, 0.05)
    with ctrl3:
        dot_size = st.slider("Dot size", 10, 80, 35, 5)

    diag_kind = "kde" if "KDE" in plot_kind else "hist"

    generate = st.button("🎨  Generate Pair Plot", use_container_width=False)

    if generate or st.session_state.get("pair_plot_generated"):
        st.session_state["pair_plot_generated"] = True

        with st.spinner("Rendering pair plot …"):
            fig = _build_pair_plot(df, features, diag_kind, scatter_alpha, dot_size)

        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)

        # Download button
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        buf.seek(0)
        st.download_button(
            "⬇️  Download Pair Plot (PNG)",
            data=buf,
            file_name="pair_plot.png",
            mime="image/png",
        )
        plt.close(fig)

        # ── Correlation heatmap ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header'><h2>Correlation Heatmap</h2>"
            "<p>Pearson correlation coefficients between all numeric features</p></div>",
            unsafe_allow_html=True,
        )

        corr = df[features].corr()
        fig2, ax = plt.subplots(figsize=(6, 4.5))
        fig2.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            linecolor=GRID_COL,
            annot_kws={"size": 10, "color": "white"},
            cbar_kws={"shrink": 0.8},
        )
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color=TEXT_COL)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=TEXT_COL)

        col_heat, col_insight = st.columns([3, 2], gap="large")
        with col_heat:
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        with col_insight:
            st.markdown(
                "<div class='section-header'><h2>Key Insights</h2>"
                "<p>Correlation with Profit</p></div>",
                unsafe_allow_html=True,
            )
            profit_corr = corr["Profit"].drop("Profit").sort_values(ascending=False)
            for feat, val in profit_corr.items():
                bar_pct = int(abs(val) * 100)
                color = "#22c55e" if val > 0.5 else ("#f59e0b" if val > 0.2 else "#ef4444")
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.8rem;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                            <span style="color:#e2e8f0; font-size:0.85rem;">{feat}</span>
                            <span style="color:{color}; font-weight:600; font-size:0.85rem;">{val:+.3f}</span>
                        </div>
                        <div style="background:#1e293b; border-radius:4px; height:8px;">
                            <div style="width:{bar_pct}%; background:{color}; border-radius:4px; height:8px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<div class='info-box' style='margin-top:1rem;'>👉 Ready to train? Go to "
            "<strong>🧪 Model Training</strong> in the sidebar.</div>",
            unsafe_allow_html=True,
        )


def _build_pair_plot(df, features, diag_kind, alpha, size):
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(11, 9))
    fig.patch.set_facecolor(DARK_BG)

    short = {
        "R&D Spend": "R&D",
        "Administration": "Admin",
        "Marketing Spend": "Marketing",
        "Profit": "Profit",
    }

    for i, row_feat in enumerate(features):
        for j, col_feat in enumerate(features):
            ax = axes[i][j]
            ax.set_facecolor(PANEL_BG)

            if i == j:
                # Diagonal
                if diag_kind == "kde":
                    sns.kdeplot(
                        data=df,
                        x=row_feat,
                        ax=ax,
                        color=PALETTE[i],
                        fill=True,
                        alpha=0.4,
                        linewidth=1.5,
                    )
                else:
                    ax.hist(
                        df[row_feat],
                        bins=12,
                        color=PALETTE[i],
                        alpha=0.7,
                        edgecolor=PANEL_BG,
                    )
            else:
                ax.scatter(
                    df[col_feat],
                    df[row_feat],
                    alpha=alpha,
                    s=size,
                    color=PALETTE[j % len(PALETTE)],
                    edgecolors="none",
                )
                # Regression line
                try:
                    m, b = np.polyfit(df[col_feat], df[row_feat], 1)
                    x_line = np.linspace(df[col_feat].min(), df[col_feat].max(), 100)
                    ax.plot(x_line, m * x_line + b, color="#f472b6", linewidth=1.2, alpha=0.8)
                except Exception:
                    pass

            # Labels on edges only
            if i == n - 1:
                ax.set_xlabel(short[col_feat], color=TEXT_COL, fontsize=8)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(short[row_feat], color=TEXT_COL, fontsize=8)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.tick_params(colors=TEXT_COL, labelsize=6.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COL)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))

    fig.suptitle("Pair Plot — Startup Features vs Profit", color="#f8fafc", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.5)
    return fig
