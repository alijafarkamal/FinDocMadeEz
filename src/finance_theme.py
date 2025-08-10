import streamlit as st

def apply_finance_theme():
    """Apply professional finance-themed styling to the Streamlit app"""
    
    finance_css = """
    <style>
    /* Finance Theme - Professional Financial Application Styling */
    
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global styling */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
        color: #e8eaed;
    }
    
    /* Header and title styling */
    .main .block-container h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: #00d4aa;
        text-shadow: 0 2px 4px rgba(0, 212, 170, 0.3);
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #00d4aa, #00b894);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main .block-container h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #00d4aa;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00d4aa;
        padding-left: 1rem;
    }
    
    .main .block-container h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #00b894;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a2332 0%, #0f1419 100%);
        border-right: 1px solid #2d3748;
    }
    
    .css-1d391kg .sidebar-content {
        background: transparent;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #00d4aa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric cards styling */
    .css-1wivap2 {
        background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .css-1wivap2:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 212, 170, 0.2);
        border-color: #00d4aa;
    }
    
    /* Metric values */
    .css-1wivap2 .metric-container {
        color: #00d4aa;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .css-1wivap2 .metric-label {
        color: #a0aec0;
        font-weight: 500;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #0f1419;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.4);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(26, 35, 50, 0.8);
        border: 2px solid #4a5568;
        border-radius: 8px;
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d4aa;
        box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.1);
        background: rgba(26, 35, 50, 0.9);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: rgba(26, 35, 50, 0.8);
        border: 2px solid #4a5568;
        border-radius: 8px;
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #00d4aa;
        box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.1);
        background: rgba(26, 35, 50, 0.9);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background: rgba(26, 35, 50, 0.8);
        border: 2px solid #4a5568;
        border-radius: 8px;
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #00d4aa;
        box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(26, 35, 50, 0.8);
        border: 1px solid #4a5568;
        border-radius: 8px;
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #0f1419;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .dataframe td {
        border-bottom: 1px solid #4a5568;
        padding: 0.75rem;
    }
    
    .dataframe tr:hover {
        background: rgba(0, 212, 170, 0.1);
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        font-family: 'Inter', sans-serif;
    }
    
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #0f1419;
        box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
    }
    
    /* Chart containers */
    .js-plotly-plot {
        background: rgba(26, 35, 50, 0.8);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4aa 0%, #00b894 100%);
        border-radius: 4px;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background: rgba(15, 20, 25, 0.9);
        border: 1px solid #4a5568;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #00d4aa 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Custom finance-themed elements */
    .finance-card {
        background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .finance-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 212, 170, 0.2);
        border-color: #00d4aa;
    }
    
    .finance-stat {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: #00d4aa;
        text-align: center;
        margin: 1rem 0;
    }
    
    .finance-label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #a0aec0;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.8rem;
    }
    
    /* Status indicators */
    .status-success {
        color: #00d4aa;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f6ad55;
        font-weight: 600;
    }
    
    .status-error {
        color: #fc8181;
        font-weight: 600;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container h1 {
            font-size: 2rem;
        }
        
        .main .block-container h2 {
            font-size: 1.5rem;
        }
        
        .finance-stat {
            font-size: 1.5rem;
        }
    }
    
    /* Loading spinner styling */
    .stSpinner > div {
        border-color: #00d4aa;
        border-top-color: transparent;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(26, 35, 50, 0.8);
        border: 2px dashed #4a5568;
        border-radius: 8px;
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #00d4aa;
        background: rgba(26, 35, 50, 0.9);
    }
    
    /* Tabs styling */
    .stTabs > div > div > div > div {
        background: rgba(26, 35, 50, 0.8);
        border: 1px solid #4a5568;
        border-radius: 8px 8px 0 0;
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    .stTabs > div > div > div > div[aria-selected="true"] {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #0f1419;
        border-color: #00d4aa;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(26, 35, 50, 0.8);
        border: 1px solid #4a5568;
        border-radius: 8px;
        color: #e8eaed;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(26, 35, 50, 0.9);
        border-color: #00d4aa;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a2332;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
    }
    
    </style>
    """
    
    st.markdown(finance_css, unsafe_allow_html=True)

def create_finance_header():
    """Create a professional finance-themed header"""
    header_html = """
    <div style="
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
        padding: 2rem 0;
        border-bottom: 2px solid #00d4aa;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
            ">
                <span style="font-size: 24px; color: #0f1419;">📊</span>
            </div>
            <h1 style="
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                font-size: 2.5rem;
                margin: 0;
                background: linear-gradient(90deg, #00d4aa, #00b894);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: 0 2px 4px rgba(0, 212, 170, 0.3);
            ">
                FinDocGPT
            </h1>
        </div>
        <p style="
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            font-size: 1.2rem;
            color: #a0aec0;
            margin: 0;
            letter-spacing: 0.5px;
        ">
            AI-Powered Financial Analysis & Investment Strategy Platform
        </p>
        <div style="
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
        ">
            <span style="
                background: rgba(0, 212, 170, 0.1);
                color: #00d4aa;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                border: 1px solid rgba(0, 212, 170, 0.3);
            ">📈 Real-time Analytics</span>
            <span style="
                background: rgba(0, 212, 170, 0.1);
                color: #00d4aa;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                border: 1px solid rgba(0, 212, 170, 0.3);
            ">🤖 AI-Powered Insights</span>
            <span style="
                background: rgba(0, 212, 170, 0.1);
                color: #00d4aa;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                border: 1px solid rgba(0, 212, 170, 0.3);
            ">💼 Portfolio Optimization</span>
        </div>
    </div>
    """
    return header_html

def create_finance_metric_card(title, value, change=None, change_type="neutral"):
    """Create a professional finance-themed metric card"""
    change_color = {
        "positive": "#00d4aa",
        "negative": "#fc8181", 
        "neutral": "#a0aec0"
    }
    
    change_icon = {
        "positive": "↗️",
        "negative": "↘️",
        "neutral": "→"
    }
    
    card_html = f"""
    <div class="finance-card" style="text-align: center;">
        <div class="finance-label">{title}</div>
        <div class="finance-stat">{value}</div>
        {f'<div style="color: {change_color[change_type]}; font-weight: 600; font-size: 0.9rem;">{change_icon[change_type]} {change}</div>' if change else ''}
    </div>
    """
    return card_html 