import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import csv
from pathlib import Path
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stacked Dashboard", page_icon= "🥞", layout="wide")

# --- CONSTANTS ---
PRIMARY_GREEN = '#2ECC71'    # Growth / Profit
BENCHMARK_ORANGE = '#E67E22' # Benchmark Line
DANGER_RED = '#E74C3C'       # Loss
NEUTRAL_GREY = '#7F8C8D'     # Invested Line
CATEGORY_COLORS = ['#F1C40F', '#95A5A6', '#3498DB', '#2ECC71', '#E67E22', '#9B59B6']
PLOTLY_TEMPLATE = 'plotly_dark'

# Registered account contribution room defaults (CAD)
TFSA_CONTRIBUTION_LIMIT = 46_700
FHSA_CONTRIBUTION_LIMIT = 24_000
RRSP_CONTRIBUTION_LIMIT = 31_560
FHSA_ANNUAL_LIMIT = 8_000

# Context for benchmarks (Added Currency Info)
BENCHMARK_CONTEXT = {
    "S&P 500": {"ticker": "^GSPC", "currency": "USD", "desc": "US Large Cap (USD)"},
    "Nasdaq-100": {"ticker": "^IXIC", "currency": "USD", "desc": "Tech Heavy (USD)"},
    "TSX Composite (Canada)": {"ticker": "^GSPTSE", "currency": "CAD", "desc": "Canadian Market (CAD)"}
}

# Hardcoded Historical Data (Fallback)
REAL_SP500_DATA = [
    3714.24, 3811.15, 3972.89, 4181.17, 4204.11, 4297.50, 4395.26, 4522.68, 4307.54, 4605.38, 
    4567.00, 4766.18, 4515.55, 4373.94, 4530.41, 4131.93, 4132.15, 3785.38, 4130.29, 3955.00, 
    3585.62, 3871.98, 4080.11, 3839.50, 4076.60, 3970.15, 4109.31, 4169.48, 4179.83, 4450.38, 
    4588.96, 4507.66, 4288.05, 4193.80, 4567.80, 4769.83, 4845.65, 5096.27, 5254.35, 5035.69, 
    5277.51, 5460.48, 5522.30, 5648.40, 5762.48, 5705.45, 6032.38, 5881.63, 6040.53, 5954.50, 
    5611.85, 5569.06, 5911.69, 6204.95, 6339.39, 6460.26, 6688.46, 6840.20, 6849.09, 6845.50, 
    6858.47
]
FALLBACK_DATES = pd.date_range(start='2021-01-01', periods=len(REAL_SP500_DATA), freq='MS')
FALLBACK_BENCH_MAP = dict(zip(FALLBACK_DATES, REAL_SP500_DATA))

TEMPLATE_CSV_PATH = Path(__file__).parent / "stacked_template.csv"

# --- 1. DATA PROCESSING FUNCTIONS ---

@st.cache_data
def get_exchange_rate():
    """Fetches USD to CAD rate (CAD=X). Returns 1.40 as fallback."""
    try:
        # Get CAD=X (amount of CAD for 1 USD)
        ticker = yf.Ticker("CAD=X")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return 1.40
    except:
        return 1.40

@st.cache_data
def get_benchmark_data(ticker_symbol, start_date):
    """Fetches real historical monthly closing prices from Yahoo Finance."""
    try:
        data = yf.download(ticker_symbol, start=start_date, interval="1mo")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[['Close']].dropna()
        data.index = data.index.to_period('M').to_timestamp()
        result = data['Close'].to_dict()
        return result if result else FALLBACK_BENCH_MAP
    except Exception:
        return FALLBACK_BENCH_MAP

@st.cache_data
def parse_data(uploaded_files):
    all_data = []

    def clean_money(s):
        if not isinstance(s, str): return 0.0
        s = s.replace('$', '').replace(',', '').replace('"', '').replace(' ', '')
        if '%' in s or s == '-' or s == '': return 0.0
        try: return float(s)
        except ValueError: return 0.0

    def classify_account(fund_name, bank_name):
        exact = str(fund_name).strip().lower()
        simple_types = {
            'tfsa': 'TFSA',
            'fhsa': 'FHSA',
            'rrsp': 'RRSP',
            'resp': 'RESP',
            'non-registered': 'Non-Registered',
        }
        if exact in simple_types:
            return simple_types[exact]

        text = (str(fund_name) + " " + str(bank_name)).lower()
        if any(k in text for k in ['first home', 'fhsa']): return 'FHSA'
        if any(k in text for k in ['education', 'resp']): return 'RESP'
        if any(k in text for k in ['tax free', 'tfsa', 'tax-free']): return 'TFSA'
        if any(k in text for k in ['rrsp', 'retirement', 'rsp', 'lira', 'locked-in']): return 'RRSP'
        if any(k in text for k in ['unregistered', 'margin', 'cash', 'individual', 'joint', 'taxable']): return 'Non-Registered'
        return 'Non-Registered'

    for uploaded_file in uploaded_files:
        try:
            stringio = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            reader = csv.reader(stringio.splitlines())
            rows = list(reader)
            current_date = None
            for row in rows:
                if not row: continue
                if len(row) > 0 and re.match(r'^\d{1,2}/\d{1,2}/\d{4}', row[0]):
                    try: current_date = pd.to_datetime(row[0])
                    except: pass
                
                market_value = 0.0
                book_cost = 0.0
                valid_row = False
                bank_name = ""
                fund_name = ""
                if current_date:
                    if len(row) >= 7:
                        market_value = clean_money(row[6])
                        book_cost = clean_money(row[3])
                        bank_name = row[1] if len(row) > 1 else ""
                        fund_name = row[2] if len(row) > 2 else ""
                        valid_row = True
                    elif len(row) == 5:
                        market_value = clean_money(row[4])
                        book_cost = clean_money(row[3])
                        bank_name = row[1] if len(row) > 1 else ""
                        fund_name = row[2] if len(row) > 2 else ""
                        valid_row = True
                    elif len(row) == 4:
                        market_value = clean_money(row[3])
                        book_cost = clean_money(row[2])
                        fund_name = row[1] if len(row) > 1 else ""
                        valid_row = True

                if valid_row and market_value > 0:
                    if fund_name != "" and "Total" not in fund_name:
                        acct_type = classify_account(fund_name, bank_name)
                        all_data.append({
                            'Date': current_date, 'Type': acct_type,
                            'Bank': bank_name, 'FundName': fund_name,
                            'Value': market_value, 'BookCost': book_cost
                        })
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")

    return pd.DataFrame(all_data).sort_values('Date') if all_data else pd.DataFrame()

def generate_example_data():
    dates = pd.date_range(start='2021-01-01', periods=len(REAL_SP500_DATA), freq='MS')
    all_records = []
    assets = [
        {"Type": "TFSA", "Monthly": 400, "Prices": REAL_SP500_DATA},
        {"Type": "FHSA", "Monthly": 150, "Prices": None, "Base": 100, "Growth": 0.002},
        {"Type": "RRSP", "Monthly": 300, "Prices": None, "Base": 100, "Growth": 0.003},
        {"Type": "RESP", "Monthly": 200, "Prices": None, "Base": 100, "Growth": 0.005},
        {"Type": "Non-Registered", "Monthly": 100, "Prices": None, "Base": 50, "Growth": 0.012, "Vol": 0.08}
    ]
    units = {a["Type"]: 0.0 for a in assets}
    prices = {a["Type"]: a.get("Base", 100) for a in assets}

    for i, date in enumerate(dates):
        for a in assets:
            acct_type = a["Type"]
            if a["Prices"]:
                price = a["Prices"][i]
            else:
                price = prices[acct_type] * (1 + (np.random.normal(a["Growth"], a.get("Vol", 0))))
                prices[acct_type] = price

            units[acct_type] += a["Monthly"] / price
            all_records.append({
                'Date': date,
                'Type': acct_type,
                'Bank': 'Demo',
                'FundName': acct_type,
                'Value': units[acct_type] * price,
                'BookCost': (i + 1) * a["Monthly"]
            })

    return pd.DataFrame(all_records)


def validate_import_data(df):
    """Return import summary and data-quality warnings for uploaded CSV data."""
    if df.empty:
        return {'summary': None, 'warnings': []}

    latest = df['Date'].max()
    n_months = df['Date'].nunique()
    fund_col = 'FundName' if 'FundName' in df.columns else 'Type'
    latest_funds = df[df['Date'] == latest][fund_col].nunique()
    summary = (
        f"Latest snapshot: {latest.strftime('%b %Y')} · "
        f"{latest_funds} accounts · {n_months} months loaded"
    )
    warnings = []

    dupes = df.duplicated(subset=['Date', fund_col], keep=False)
    if dupes.any():
        warnings.append(
            f"Duplicate rows for the same date and fund ({int(dupes.sum())} rows)."
        )

    dates = sorted(df['Date'].unique())
    if len(dates) >= 2:
        gap_msgs = []
        for i in range(1, len(dates)):
            prev, curr = pd.Timestamp(dates[i - 1]), pd.Timestamp(dates[i])
            month_gap = (curr.year - prev.year) * 12 + (curr.month - prev.month)
            if month_gap > 1:
                gap_msgs.append(
                    f"{prev.strftime('%b %Y')} → {curr.strftime('%b %Y')} "
                    f"({month_gap - 1} mo gap)"
                )
        if gap_msgs:
            extra = "..." if len(gap_msgs) > 3 else ""
            warnings.append("Missing months: " + "; ".join(gap_msgs[:3]) + extra)

    for fund, grp in df.groupby(fund_col):
        by_date = grp.groupby('Date')['BookCost'].sum().sort_index()
        if len(by_date) < 2:
            continue
        if (by_date.diff().dropna() < -0.01).any():
            warnings.append(
                f"Book Cost decreased for {fund} — may reflect a withdrawal or data change."
            )

    return {'summary': summary, 'warnings': warnings}


def render_import_feedback(df):
    """Show import summary and validation warnings."""
    feedback = validate_import_data(df)
    if feedback['summary']:
        st.caption(feedback['summary'])
    for msg in feedback['warnings']:
        st.warning(msg)

# --- 2. MONTE CARLO SIMULATOR ---

def run_monte_carlo(current_val, monthly_add, years, mean_ret_pct, vol_pct, num_sims=500):
    """Generates future portfolio paths based on geometric brownian motion."""
    months = int(years * 12)
    mu = mean_ret_pct / 12  # Monthly return
    sigma = vol_pct / np.sqrt(12) # Monthly volatility
    
    # Random shocks: matrix of shape [months, num_sims]
    shocks = np.random.normal(mu - 0.5 * sigma**2, sigma, (months, num_sims))
    
    # Initialize paths
    paths = np.zeros((months + 1, num_sims))
    paths[0] = current_val
    
    for t in range(1, months + 1):
        # Growth step
        growth = np.exp(shocks[t-1])
        paths[t] = paths[t-1] * growth
        # Contribution step
        paths[t] += monthly_add
        
    return paths

# --- CONTRIBUTION LIMIT HELPERS ---

def get_contribution_usage(df, account_type):
    """Sum BookCost for an account type at the latest snapshot (CAD source data)."""
    if df.empty:
        return 0.0
    latest_date = df['Date'].max()
    mask = (df['Date'] == latest_date) & (df['Type'] == account_type)
    return df.loc[mask, 'BookCost'].sum()


def get_ytd_contribution(df, account_type):
    """Estimate calendar-year contributions from BookCost changes (CAD source data)."""
    if df.empty:
        return 0.0
    latest_date = df['Date'].max()
    current_year = latest_date.year
    type_df = df[df['Type'] == account_type]
    if type_df.empty:
        return 0.0

    in_year = type_df[type_df['Date'].dt.year == current_year]
    if in_year.empty:
        return 0.0

    latest_snapshot = in_year[in_year['Date'] == in_year['Date'].max()]['BookCost'].sum()
    prior = type_df[type_df['Date'].dt.year < current_year]
    baseline = 0.0
    if not prior.empty:
        last_prior_date = prior['Date'].max()
        baseline = prior[prior['Date'] == last_prior_date]['BookCost'].sum()

    return max(latest_snapshot - baseline, 0.0)


def avg_monthly_contribution(df, account_type):
    """Average monthly BookCost increase for an account type."""
    type_df = df[df['Type'] == account_type]
    if type_df.empty:
        return 0.0
    by_date = type_df.groupby('Date')['BookCost'].sum().sort_index()
    if len(by_date) < 2:
        return 0.0
    return by_date.diff().dropna().mean()


def contribution_pct(used, limit):
    """Return percentage of contribution limit used (uncapped for over-limit display)."""
    return (used / limit * 100) if limit > 0 else 0.0


def limit_bar_color(pct):
    if pct > 100:
        return DANGER_RED
    if pct >= 90:
        return BENCHMARK_ORANGE
    return PRIMARY_GREEN


LABEL_GREY = 'rgba(255, 255, 255, 0.45)'
CRA_LIMIT_INFO = (
    "<b>RRSP:</b> Your deduction limit is on your account dashboard.<br><br>"
    "<b>TFSA:</b> Available room is listed here, but CRA updates it once a year. "
    "Check from <b>April</b> for the previous year's records.<br><br>"
    "<b>FHSA:</b> Deduction limits and participation room are in your CRA profile."
)


def render_cra_info_tooltip():
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"] {{
            overflow-x: hidden !important;
        }}
        .cra-info-wrap {{
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }}
        .cra-info-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.6rem;
            height: 1.6rem;
            border-radius: 50%;
            border: 1px solid #7F8C8D;
            color: #7F8C8D;
            font-size: 0.8rem;
            font-weight: 700;
            font-style: italic;
            font-family: Georgia, serif;
            cursor: help;
            user-select: none;
            background: transparent;
        }}
        .cra-info-wrap .cra-info-tip {{
            visibility: hidden;
            opacity: 0;
            position: absolute;
            right: 0;
            bottom: calc(100% + 8px);
            width: 260px;
            max-width: 85vw;
            padding: 12px 14px;
            border-radius: 8px;
            background: #1e1e1e;
            color: #fafafa;
            font-size: 0.82rem;
            line-height: 1.45;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
            border: 1px solid #333;
            transition: opacity 0.15s ease;
            z-index: 9999;
            pointer-events: none;
            text-align: left;
        }}
        .cra-info-wrap:hover .cra-info-tip {{
            visibility: visible;
            opacity: 1;
        }}
        </style>
        <div class="cra-info-wrap">
            <span class="cra-info-btn" title="CRA limit lookup tips">i</span>
            <div class="cra-info-tip">{CRA_LIMIT_INFO}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

CRA_MY_ACCOUNT_URL = (
    "https://www.canada.ca/en/revenue-agency/services/e-services/"
    "digital-services-individuals/account-individuals.html"
)
ACCOUNT_FULL_NAMES = {
    'TFSA': 'Tax-Free Savings Account',
    'FHSA': 'First Home Savings Account',
    'RRSP': 'Registered Retirement Savings Plan',
}


def render_account_gauge(pct):
    """Semi-circular gauge for a single account's limit usage."""
    color = limit_bar_color(pct)
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pct,
        number={
            'suffix': '% USED',
            'font': {'size': 28, 'color': color},
            'valueformat': '.1f',
        },
        gauge={
            'shape': 'angular',
            'axis': {
                'range': [0, 100],
                'tickmode': 'array',
                'tickvals': [0, 20, 40, 60, 80, 100],
                'ticktext': ['0', '20', '40', '60', '80', '100'],
                'tickwidth': 0,
                'tickcolor': NEUTRAL_GREY,
            },
            'bar': {'color': color, 'thickness': 0.85},
            'bgcolor': 'rgba(149, 165, 166, 0.2)',
            'borderwidth': 0,
            'steps': [{'range': [0, 100], 'color': 'rgba(149, 165, 166, 0.15)'}],
        },
    ))
    fig.update_layout(
        height=190,
        margin=dict(l=40, r=70, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def format_limit_line(label, value, italic_fragment=None):
    """Detail row with muted label and bold dollar value."""
    if italic_fragment and italic_fragment in label:
        label_html = label.replace(
            italic_fragment, f"<i>{italic_fragment}</i>", 1
        )
    else:
        label_html = label
    return (
        f"<span class='limit-label'>{label_html}</span> "
        f"<span class='limit-value'>${value:,.0f}</span>"
    )


def render_account_details(acct):
    """Centered summary lines below each gauge card."""
    if acct['label'] == 'FHSA':
        title_label = "Total Remaining Room"
    else:
        title_label = "Remaining Room"

    has_ytd = acct.get('ytd_used') is not None and acct.get('annual_limit') is not None
    line_style = "text-align:center;margin:6px 0;min-height:1.75rem;line-height:1.75rem"
    title_html = (
        f"<span class='limit-value limit-card-title-value'>${acct['remaining']:,.0f}</span> "
        f"<span class='limit-label limit-card-title-label'>{title_label}</span>"
    )

    if has_ytd:
        ytd_rem = max(acct['annual_limit'] - acct['ytd_used'], 0.0)
        stats_lines = (
            f"<p style='{line_style}'>{format_limit_line('Remaining Room this Year:', ytd_rem, italic_fragment='this Year')}</p>"
            f"<p style='{line_style}'>{format_limit_line('Lifetime Limit:', acct['limit'])}</p>"
            f"<p style='{line_style}'>{format_limit_line('Total Contributions:', acct['used'])}</p>"
        )
    else:
        stats_lines = (
            f"<p style='{line_style}' class='limit-card-stats-spacer' aria-hidden='true'>&nbsp;</p>"
            f"<p style='{line_style}'>{format_limit_line('Contribution Limit:', acct['limit'])}</p>"
            f"<p style='{line_style}'>{format_limit_line('Total Contributions:', acct['used'])}</p>"
        )

    st.markdown(
        f"<p class='limit-card-title' style='text-align:center;width:100%;display:flex;"
        f"justify-content:center;align-items:center;flex-wrap:wrap;gap:0.35rem;margin:0 0 12px 0'>"
        f"{title_html}</p>"
        f"<div class='limit-card-stats'>{stats_lines}</div>",
        unsafe_allow_html=True,
    )


def render_contribution_limits_section(accounts):
    """Card-based gauge layout for registered contribution room."""
    for acct in accounts:
        acct['remaining'] = max(acct['limit'] - acct['used'], 0.0)
        acct['pct'] = contribution_pct(acct['used'], acct['limit'])
        acct['full_name'] = ACCOUNT_FULL_NAMES.get(acct['label'], acct['label'])

    st.markdown(
        """
        <style>
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            align-items: stretch !important;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            display: flex !important;
            flex-direction: column !important;
            align-self: stretch !important;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
            width: 100%;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
            min-height: 540px;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] > div {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] [data-testid="stPlotlyChart"] {
            margin-bottom: -1.25rem;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] {
            text-align: center !important;
            width: 100%;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-card-header {
            min-height: 4rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center !important;
            margin-bottom: 4px;
            width: 100%;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-card-title {
            text-align: center !important;
            margin: 0 0 12px 0;
            min-height: 3.5rem;
            display: flex !important;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.35rem;
            font-size: 1.05rem;
            width: 100%;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-label {
            color: #7F8C8D;
            font-weight: 400;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-value {
            color: inherit;
            font-weight: 700;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-card-title-value {
            font-size: 1.15rem;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-card-stats {
            display: flex;
            flex-direction: column;
            width: 100%;
            min-height: 8.75rem;
            flex: 1 1 auto;
        }
        #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-card-stats-spacer {
            visibility: hidden;
        }
        @media (prefers-color-scheme: dark) {
            #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-label {
                color: rgba(255, 255, 255, 0.45);
            }
            #limit-cards-anchor + div[data-testid="stHorizontalBlock"] .limit-value {
                color: #ffffff;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<span id="limit-cards-anchor"></span>', unsafe_allow_html=True)

    cols = st.columns(len(accounts))
    for col, acct in zip(cols, accounts):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div class='limit-card-header' style='text-align:center;width:100%;"
                    f"display:flex;align-items:center;justify-content:center'>"
                    f"<b>{acct['label']}</b> — {acct['full_name']}</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    render_account_gauge(acct['pct']),
                    use_container_width=True,
                    key=f"gauge_{acct['label']}",
                )
                render_account_details(acct)

                if acct['used'] > acct['limit']:
                    st.error(f"Over limit by ${acct['used'] - acct['limit']:,.0f}")
                elif acct['pct'] >= 90:
                    st.warning(f"Approaching limit ({acct['pct']:.0f}% used)")

# --- 3. VISUALIZATION ENGINE ---

def render_dashboard(df, bench_key, bench_info, target_currency, usd_cad_rate, contribution_limits):
    if df.empty:
        st.warning("No data available to render.")
        return

    # --- CURRENCY NORMALIZATION LOGIC ---
    # 1. Determine User Conversion Factors (Assuming User Data is CAD)
    user_fx_multiplier = 1.0
    if target_currency == "USD":
        user_fx_multiplier = 1.0 / usd_cad_rate

    # 2. Benchmark FX Logic
    bench_currency = bench_info['currency']
    bench_fx_multiplier = 1.0
    
    if bench_currency == "USD" and target_currency == "CAD":
        bench_fx_multiplier = usd_cad_rate
    elif bench_currency == "CAD" and target_currency == "USD":
        bench_fx_multiplier = 1.0 / usd_cad_rate

    # 3. Apply Conversion to Data
    df_conv = df.copy()
    df_conv['Value'] = df_conv['Value'] * user_fx_multiplier
    df_conv['BookCost'] = df_conv['BookCost'] * user_fx_multiplier

    # --- AGGREGATION ---
    df_total = df_conv.groupby('Date')[['Value', 'BookCost']].sum().reset_index()
    unique_types = sorted(df_conv['Type'].unique())
    color_map = {t: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, t in enumerate(unique_types)}
    
    # --- BENCHMARK CALCULATION ---
    bench_data_raw = get_benchmark_data(bench_info['ticker'], df_total['Date'].min())
    if not bench_data_raw:
        bench_data_raw = FALLBACK_BENCH_MAP

    bench_units = 0
    bench_values = []
    first_date_match = next(iter(bench_data_raw), None)
    first_price_raw = bench_data_raw.get(first_date_match, 100.0) if first_date_match else 100.0
    
    df_total['Incr_Invest'] = df_total['BookCost'].diff().fillna(df_total['BookCost'].iloc[0])
    
    for _, row in df_total.iterrows():
        match_date = row['Date'].replace(day=1) # normalize to start of month
        # Get raw price and apply FX
        raw_price = bench_data_raw.get(match_date, first_price_raw)
        adj_price = raw_price * bench_fx_multiplier
        
        bench_units += row['Incr_Invest'] / adj_price
        bench_values.append(bench_units * adj_price)
    
    df_total['BenchValue'] = bench_values
    latest = df_total.iloc[-1]
    
    # --- KPI Calculations ---
    u_roi = (latest['Value'] / latest['BookCost'] - 1) * 100 if latest['BookCost'] > 0 else 0
    b_roi = (latest['BenchValue'] / latest['BookCost'] - 1) * 100 if latest['BookCost'] > 0 else 0

    # --- TABS LAYOUT ---
    tab1, tab2 = st.tabs(["Portfolio Performance", "Future Simulator (FIRE)"])

    # --- TAB 1: HISTORY ---
    with tab1:
        # 1. Metric Row
        gain_dollar = latest['Value'] - latest['BookCost']
        gain_pct = u_roi
        gain_color = PRIMARY_GREEN if gain_dollar >= 0 else DANGER_RED
        if gain_dollar >= 0:
            gain_line = f"+${gain_dollar:,.2f} (+{gain_pct:.2f}%)"
        else:
            gain_line = f"-${abs(gain_dollar):,.2f} ({gain_pct:.2f}%)"
        st.markdown(
            f'<p style="color:#7F8C8D;font-size:0.95rem;margin:0 0 0.35rem 0">'
            f'Net Worth ({target_currency})</p>'
            f'<p style="font-size:2.75rem;font-weight:700;margin:0;line-height:1.1">'
            f'${latest["Value"]:,.2f}</p>'
            f'<p style="color:{gain_color};font-size:1.1rem;font-weight:600;'
            f'margin:0.35rem 0 1.25rem 0">{gain_line}</p>',
            unsafe_allow_html=True,
        )
        monthly_contrib = (
            df_total['BookCost'].diff().dropna().mean()
            if len(df_total) >= 2 else 0.0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Invested", f"${latest['BookCost']:,.2f}")
        c2.metric("Monthly Contribution (avg)", f"${monthly_contrib:,.2f}")
        c3.metric(f"{bench_key} ROI", f"{b_roi:.2f}%", delta=f"{(u_roi - b_roi):.2f}% vs Market")

        st.markdown("---")

        # 2. Chart: Net Worth vs Invested
        st.subheader("Net Worth vs. Invested (Capital Growth)")
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['BookCost'], 
            mode='lines', name='Invested', 
            line=dict(color=NEUTRAL_GREY, dash='dash'), 
            hovertemplate='Invested: $%{y:,.2f}'
        ))
        fig_net.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['Value'], 
            mode='lines', name='Net Worth', 
            fill='tonexty', line=dict(color=PRIMARY_GREEN), 
            hovertemplate='Value: $%{y:,.2f}'
        ))
        fig_net.update_layout(
            hovermode="x unified", template=PLOTLY_TEMPLATE, height=400, 
            yaxis_tickprefix="$", yaxis_tickformat=",.2f"
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # 3. Chart: Benchmark Comparison
        st.subheader(f"Portfolio vs. {bench_key}")
        st.info(f"Comparing your Active Strategy vs. Passive {bench_key} Index Fund.")

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['BenchValue'], 
            name=f'{bench_key} ({target_currency})', 
            line=dict(color=BENCHMARK_ORANGE, width=2), 
            hovertemplate=f'{bench_key}: '+'$%{y:,.2f}'
        ))
        fig_bench.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['Value'], 
            name='My Portfolio', 
            line=dict(color=PRIMARY_GREEN, width=3), 
            hovertemplate='My Portfolio: $%{y:,.2f}'
        ))
        fig_bench.update_layout(
            hovermode="x unified", template=PLOTLY_TEMPLATE, height=400, 
            yaxis_tickprefix="$", yaxis_tickformat=",.2f"
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        # 4. Charts: Composition & Returns
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Portfolio Composition")
            fig_comp = px.area(
                df_conv.groupby(['Date', 'Type'])['Value'].sum().reset_index(),
                x="Date", y="Value", color="Type",
                color_discrete_map=color_map
            )
            fig_comp.update_layout(
                hovermode="x unified", template=PLOTLY_TEMPLATE, 
                yaxis_tickprefix="$", yaxis_tickformat=",.2f"
            )
            fig_comp.update_traces(hovertemplate='$%{y:,.2f}')
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_b:
            st.subheader("Monthly Market Gain ($)")
            df_total['TotalDiff'] = df_total['Value'].diff().fillna(0)
            df_total['Contribution'] = df_total['BookCost'].diff().fillna(0)
            df_total['MarketGain'] = df_total['TotalDiff'] - df_total['Contribution']
            df_total['BarColor'] = df_total['MarketGain'].apply(lambda x: PRIMARY_GREEN if x >= 0 else DANGER_RED)
            
            fig_bar = go.Figure(go.Bar(
                x=df_total['Date'], y=df_total['MarketGain'], 
                marker_color=df_total['BarColor'], 
                hovertemplate='Gain: $%{y:,.2f}<extra></extra>'
            ))
            fig_bar.update_layout(
                template=PLOTLY_TEMPLATE, 
                yaxis_tickprefix="$", yaxis_tickformat=",.2f"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # 5. Chart: Allocation Donut
        st.subheader("Current Allocation")
        latest_date = df_conv['Date'].max()
        fig_pie = px.pie(
            df_conv[df_conv['Date'] == latest_date].groupby('Type')['Value'].sum().reset_index(),
            values='Value', names='Type', hole=0.5,
            color='Type', color_discrete_map=color_map
        )

        fig_pie.update_traces(
            textfont_color='white', textinfo='percent+label',
            insidetextorientation='horizontal',
            hovertemplate='%{label}<br>$%{value:,.2f}<br>%{percent}'
        )

        fig_pie.update_layout(
            annotations=[dict(text=f"${latest['Value']:,.2f}", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # 6. Registered Contribution Limits
        st.subheader("Registered Contribution Limits")
        st.caption(
            "These figures are estimates based on Book Cost, a proxy for contributed capital. "
            "Withdrawals, in-kind transfers, or cost-base adjustments may differ from CRA records."
        )

        tfsa_used = get_contribution_usage(df, 'TFSA')
        fhsa_used = get_contribution_usage(df, 'FHSA')
        rrsp_used = get_contribution_usage(df, 'RRSP')
        fhsa_ytd = get_ytd_contribution(df, 'FHSA')

        render_contribution_limits_section([
            {'label': 'TFSA', 'used': tfsa_used, 'limit': contribution_limits['tfsa']},
            {
                'label': 'FHSA', 'used': fhsa_used, 'limit': contribution_limits['fhsa'],
                'ytd_used': fhsa_ytd, 'annual_limit': FHSA_ANNUAL_LIMIT,
            },
            {'label': 'RRSP', 'used': rrsp_used, 'limit': contribution_limits['rrsp']},
        ])

        with st.expander("View Raw Data"):
            st.dataframe(df_conv)

    # --- TAB 2: SIMULATOR ---
    with tab2:
        st.subheader("Monte Carlo Wealth Projection")
        st.info("This simulation runs 500 possible market scenarios to estimate your future net worth.")
        
        # Controls
        col_input, col_chart = st.columns([1, 3])
        
        with col_input:
            st.markdown("### Parameters")
            sim_years = st.slider("Years to Grow", 1, 60, 25)
            sim_contrib = st.number_input(f"Monthly Contribution ({target_currency})", value=2000, step=100)
            sim_return = st.slider("Exp. Annual Return (%)", 0.0, 50.0, 7.0) / 100
            sim_vol = st.slider("Volatility (%)", 5.0, 30.0, 15.0) / 100
            
            current_nw = latest['Value']
            st.divider()
            st.metric("Starting Capital", f"${current_nw:,.2f}")

            tfsa_monthly = avg_monthly_contribution(df, 'TFSA')
            fhsa_monthly = avg_monthly_contribution(df, 'FHSA')
            tfsa_room = max(contribution_limits['tfsa'] - get_contribution_usage(df, 'TFSA'), 0)
            fhsa_room = max(contribution_limits['fhsa'] - get_contribution_usage(df, 'FHSA'), 0)
            if tfsa_monthly > 0 and tfsa_room > 0:
                st.caption(f"TFSA room fills in ~{tfsa_room / tfsa_monthly:.0f} mo at ${tfsa_monthly:,.0f}/mo avg")
            if fhsa_monthly > 0 and fhsa_room > 0:
                st.caption(f"FHSA room fills in ~{fhsa_room / fhsa_monthly:.0f} mo at ${fhsa_monthly:,.0f}/mo avg")
        
        # Run Simulation
        paths = run_monte_carlo(current_nw, sim_contrib, sim_years, sim_return, sim_vol)
        
        # Calculate Percentiles
        p10 = np.percentile(paths, 10, axis=1)
        p50 = np.percentile(paths, 50, axis=1)
        p90 = np.percentile(paths, 90, axis=1)
        
        future_dates = pd.date_range(start=latest['Date'], periods=len(p50), freq='ME')
        
        # Plotting
        with col_chart:
            fig_mc = go.Figure()
            # 90th Percentile (Optimistic)
            fig_mc.add_trace(go.Scatter(
                x=future_dates, y=p90, mode='lines', line=dict(color=BENCHMARK_ORANGE,width=3), name='Upper Bound (90%)', hovertemplate="$%{y:,.0f}"
            ))
            # 10th Percentile (Pessimistic) - Fill area
            fig_mc.add_trace(go.Scatter(
                x=future_dates, y=p10, mode='lines', line=dict(color= DANGER_RED, width=3), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)', name='Lower Bound (10%)', hovertemplate="$%{y:,.0f}"
            ))
            # Median
            fig_mc.add_trace(go.Scatter(
                x=future_dates, y=p50, mode='lines', line=dict(color=PRIMARY_GREEN, width=3), name='Median Outcome' , hovertemplate="$%{y:,.0f}"
            ))
            
            final_median = p50[-1]
            fig_mc.update_layout(
                title=f"Projected Median (In {sim_years} Years): ${final_median:,.2f}", 
                template=PLOTLY_TEMPLATE, hovermode="x unified", yaxis_tickprefix="$"
            )
            st.plotly_chart(fig_mc, use_container_width=True)

# --- 4. MAIN CONTROLLER ---

# Initialize Session State
if 'demo_active' not in st.session_state:
    st.session_state.demo_active = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'contribution_limits' not in st.session_state:
    st.session_state.contribution_limits = {
        'tfsa': TFSA_CONTRIBUTION_LIMIT,
        'fhsa': FHSA_CONTRIBUTION_LIMIT,
        'rrsp': RRSP_CONTRIBUTION_LIMIT,
    }

st.title("🥞Stacked: Investment Dashboard")

# Initialize FX
usd_cad_rate = get_exchange_rate()

with st.sidebar:
    st.header("1. Data Import")
    uploaded_files = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    
    col_demo, col_clear = st.columns(2)
    with col_demo:
        if st.button("Load Demo"):
            st.session_state.demo_active = True
            st.session_state.uploader_key += 1 # Increment key to reset uploader
            st.rerun()
    with col_clear:
        if st.button("Clear Data"):
            st.session_state.demo_active = False
            st.session_state.uploader_key += 1 # Increment key to reset uploader
            st.rerun()

    if uploaded_files:
        st.session_state.demo_active = False
        render_import_feedback(parse_data(uploaded_files))

    with st.expander("Template CSV"):
        st.caption(
            "Download a 12-month sample. Use TFSA, FHSA, RRSP, RESP, or Non-Registered as the fund name."
        )
        template = TEMPLATE_CSV_PATH.read_text(encoding="utf-8")
        st.download_button("Download Template", template, "stacked_template.csv", "text/csv")

    st.markdown("---")

    # 2. Settings
    st.header("2. Settings")

    # Currency Toggle
    target_currency = st.radio("Display Currency", ["CAD", "USD"], horizontal=True)
    st.caption(f"Live Rate: 1 USD = {usd_cad_rate:.2f} CAD")

    # Benchmark Selector
    bench_choice = st.selectbox("Compare Against:", list(BENCHMARK_CONTEXT.keys()))
    st.caption(BENCHMARK_CONTEXT[bench_choice]['desc'])

    st.markdown("---")

    # 3. Contribution Room
    st.header("3. Contribution Room (CAD)")
    limits = st.session_state.contribution_limits
    contribution_limits = {
        'tfsa': st.number_input(
            "TFSA Lifetime Room", value=limits['tfsa'], step=500, min_value=0,
        ),
        'fhsa': st.number_input(
            "FHSA Lifetime Room", value=limits['fhsa'], step=500, min_value=0,
        ),
        'rrsp': st.number_input(
            "RRSP Room", value=limits['rrsp'], step=500, min_value=0,
        ),
    }
    st.session_state.contribution_limits = contribution_limits
    cra_col, info_col = st.columns([8, 1])
    with cra_col:
        st.link_button("CRA My Account", CRA_MY_ACCOUNT_URL, use_container_width=True)
    with info_col:
        render_cra_info_tooltip()


# Logic to Switch Data Source
if st.session_state.demo_active:
    df_to_show = generate_example_data()
    render_dashboard(df_to_show, bench_choice, BENCHMARK_CONTEXT[bench_choice], target_currency, usd_cad_rate, contribution_limits)
elif uploaded_files:
    df_to_show = parse_data(uploaded_files)
    render_dashboard(df_to_show, bench_choice, BENCHMARK_CONTEXT[bench_choice], target_currency, usd_cad_rate, contribution_limits)
else:
    st.info("Upload your CSV files to begin, or click 'Load Demo' in the sidebar.")
