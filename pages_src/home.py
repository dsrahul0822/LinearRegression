import streamlit as st


def render():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📈 Profit <span class="accent">Predictor</span></h1>
            <p>A professional Multiple Linear Regression suite for startup profit forecasting.<br>
            Upload your data, explore insights, train the model, and generate predictions — all in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Workflow steps ────────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'><h2>How It Works</h2>"
        "<p>Follow the four-step workflow using the sidebar navigation</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1", "📂 Upload Data",       "Import your CSV dataset and preview the raw records."),
        ("2", "📊 Data Analysis",     "Visualise feature relationships with an interactive pair plot."),
        ("3", "🧪 Model Training",    "Split data, train the model, and evaluate performance metrics."),
        ("4", "🔮 Predict Profit",    "Enter new values and get an instant profit prediction."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{num}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── About the model ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(
            "<div class='section-header'><h2>About the Model</h2>"
            "<p>Multiple Linear Regression — interpretable, fast, and effective</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-box">
            This application uses <strong>Multiple Linear Regression</strong> to model the linear
            relationship between a startup's operational spending and its net profit.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            **Features used for prediction:**
            - **R&D Spend** — investment in research and development
            - **Administration** — administrative and operational costs
            - **Marketing Spend** — budget allocated to marketing activities

            **Target variable:**
            - **Profit** — the net profit earned by the startup

            The model is trained using *scikit-learn's LinearRegression* and evaluated on a
            held-out test set using five industry-standard regression metrics.
            """,
        )

    with col_right:
        st.markdown(
            "<div class='section-header'><h2>Quick Stats</h2>"
            "<p>Dataset overview</p></div>",
            unsafe_allow_html=True,
        )
        stats = [
            ("50", "Startup Records", "Total rows in dataset"),
            ("3",  "Predictors",      "R&D · Admin · Marketing"),
            ("5",  "Metrics",         "R² · MAE · MAPE · MSE · RMSE"),
        ]
        for val, label, sub in stats:
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom:0.8rem;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='warning-box'>👉 <strong>Get started:</strong> Click <em>📂 Upload Data</em> in the sidebar to begin.</div>",
        unsafe_allow_html=True,
    )
