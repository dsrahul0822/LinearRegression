import streamlit as st
import pandas as pd


REQUIRED_COLS = {"R&D Spend", "Administration", "Marketing Spend", "Profit"}


def render():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📂 Upload <span class="accent">Dataset</span></h1>
            <p>Import your startup CSV file to begin the analysis pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Upload widget ─────────────────────────────────────────────────────────
    col_up, col_info = st.columns([2, 1], gap="large")

    with col_up:
        st.markdown(
            "<div class='section-header'><h2>Select File</h2>"
            "<p>Accepted format: CSV (.csv)</p></div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Drop your CSV here or click to browse",
            type=["csv"],
            help="The file must contain columns: R&D Spend, Administration, Marketing Spend, Profit",
        )

    with col_info:
        st.markdown(
            "<div class='section-header'><h2>Requirements</h2>"
            "<p>File specifications</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-box">
            <strong>Required columns:</strong><br>
            • R&D Spend<br>
            • Administration<br>
            • Marketing Spend<br>
            • Profit<br><br>
            <strong>Optional columns</strong> (ignored):<br>
            • State or any other columns
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Process upload ────────────────────────────────────────────────────────
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            st.markdown(
                f"<div class='warning-box'>⚠️ Missing columns: <strong>{', '.join(missing)}</strong>. "
                "Please check your file and re-upload.</div>",
                unsafe_allow_html=True,
            )
            return

        # Keep only relevant numeric columns
        features = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
        df_model = df[features].copy()

        # Drop rows with nulls
        before = len(df_model)
        df_model.dropna(inplace=True)
        after = len(df_model)

        # Save to session state
        st.session_state["df"] = df_model
        # Clear downstream artefacts when new data is loaded
        for key in ["model", "X_test", "y_test", "y_pred", "metrics", "split_ratio"]:
            st.session_state.pop(key, None)

        st.markdown(
            f"<div class='success-box'>✅ Dataset loaded successfully — "
            f"<strong>{after} rows</strong> × <strong>{len(features)} columns</strong>"
            + (f" ({before - after} rows with missing values were dropped)." if before != after else ".")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Tabs: Preview / Statistics ────────────────────────────────────────
        tab_prev, tab_stats, tab_dist = st.tabs(
            ["  📋  Data Preview  ", "  📐  Descriptive Statistics  ", "  📊  Column Info  "]
        )

        with tab_prev:
            st.markdown("<br>", unsafe_allow_html=True)
            rows = st.slider("Rows to display", 5, min(50, after), min(10, after), key="preview_rows")
            st.dataframe(
                df_model.head(rows).style.format("${:,.2f}").background_gradient(
                    subset=["Profit"], cmap="Blues"
                ),
                use_container_width=True,
            )

        with tab_stats:
            st.markdown("<br>", unsafe_allow_html=True)
            desc = df_model.describe().T
            desc.index.name = "Feature"
            st.dataframe(
                desc.style.format("{:,.2f}").background_gradient(cmap="Blues"),
                use_container_width=True,
            )

        with tab_dist:
            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b, col_c, col_d = st.columns(4)
            for col, feat in zip([col_a, col_b, col_c, col_d], features):
                with col:
                    mn  = df_model[feat].min()
                    mx  = df_model[feat].max()
                    avg = df_model[feat].mean()
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{feat}</div>
                            <div class="metric-value" style="font-size:1.2rem;">${avg:,.0f}</div>
                            <div class="metric-sub">avg</div>
                            <div class="metric-sub">min: ${mn:,.0f} | max: ${mx:,.0f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown(
            "<div class='info-box' style='margin-top:1.2rem;'>👉 Proceed to "
            "<strong>📊 Data Analysis</strong> in the sidebar to explore feature relationships.</div>",
            unsafe_allow_html=True,
        )

    else:
        if "df" in st.session_state:
            st.markdown(
                "<div class='success-box'>✅ A dataset is already loaded in this session. "
                "Navigate to other pages or re-upload to replace it.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='info-box'>ℹ️ No file uploaded yet. Use the uploader above to get started.</div>",
                unsafe_allow_html=True,
            )
