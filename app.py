import streamlit as st

st.set_page_config(
    page_title="Profit Predictor | The Scholar",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 6px 0;
    }

    /* Main background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 60%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .hero-banner h1 {
        color: #f8fafc !important;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .hero-banner p {
        color: #94a3b8 !important;
        font-size: 1.05rem;
    }
    .hero-banner .accent { color: #38bdf8 !important; }

    /* Step cards on home */
    .step-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        height: 100%;
    }
    .step-card .step-num {
        background: linear-gradient(135deg, #3b82f6, #0ea5e9);
        color: white;
        border-radius: 50%;
        width: 38px; height: 38px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 1rem;
        margin: 0 auto 0.8rem auto;
    }
    .step-card h3 { color: #f1f5f9 !important; font-size: 1rem; margin-bottom: 0.4rem; }
    .step-card p  { color: #94a3b8 !important; font-size: 0.85rem; }

    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .metric-label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-card .metric-value {
        color: #38bdf8;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-card .metric-sub {
        color: #475569;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-header {
        border-left: 4px solid #3b82f6;
        padding-left: 0.8rem;
        margin-bottom: 1rem;
    }
    .section-header h2 { color: #f1f5f9 !important; font-size: 1.3rem; font-weight: 600; }
    .section-header p  { color: #64748b !important; font-size: 0.88rem; }

    /* Info / warning boxes */
    .info-box {
        background: #0c1e35;
        border: 1px solid #1d4ed8;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #93c5fd !important;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background: #052e16;
        border: 1px solid #166534;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #86efac !important;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: #1c1003;
        border: 1px solid #92400e;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #fcd34d !important;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Prediction result */
    .prediction-result {
        background: linear-gradient(135deg, #0c1e35, #0f2744);
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .prediction-result .pred-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .prediction-result .pred-value {
        color: #38bdf8;
        font-size: 3rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .prediction-result .pred-sub {
        color: #475569;
        font-size: 0.85rem;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.8rem;
        font-size: 0.95rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    /* Slider */
    .stSlider > div > div > div > div { background: #3b82f6 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 8px 8px 0 0;
        color: #94a3b8 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important;
        color: white !important;
    }

    /* Number input */
    .stNumberInput input {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
    }

    /* Divider */
    hr { border-color: #1e293b !important; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
            <div style='font-size:2.2rem;'>📈</div>
            <div style='font-size:1.1rem; font-weight:700; color:#f8fafc;'>Profit Predictor</div>
            <div style='font-size:0.78rem; color:#475569; margin-top:2px;'>The Scholar • ML Suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "📂  Upload Data",
            "📊  Data Analysis",
            "🧪  Model Training",
            "🔮  Predict Profit",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#334155; padding:0.5rem 0;'>"
        "Model: Multiple Linear Regression<br>"
        "Framework: scikit-learn<br>"
        "Version: 1.0.0"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Route Pages ───────────────────────────────────────────────────────────────
if page == "🏠  Home":
    from pages_src.home import render
elif page == "📂  Upload Data":
    from pages_src.upload import render
elif page == "📊  Data Analysis":
    from pages_src.analysis import render
elif page == "🧪  Model Training":
    from pages_src.training import render
elif page == "🔮  Predict Profit":
    from pages_src.predict import render

render()
