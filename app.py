import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4CAF50;
        margin: 8px 0;
    }
    .metric-card-red { border-left: 4px solid #f44336; }
    .metric-card-blue { border-left: 4px solid #2196F3; }
    .metric-card-orange { border-left: 4px solid #FF9800; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2196F3;
    }
    .stSelectbox label { color: #b0bec5 !important; }
    .sidebar .sidebar-content { background-color: #1a1d2e; }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2d3250;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def find_file(*candidates):
    """Find a file by trying multiple path candidates."""
    import os
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

@st.cache_data
def load_data(uploaded_file=None):
    meta_path = find_file('stock_metadata.csv', 'data/stock_metadata.csv')
    if not meta_path:
        st.error("❌ stock_metadata.csv not found.")
        st.stop()
    meta = pd.read_csv(meta_path)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    else:
        stock_path = find_file('NIFTY50_all.csv', 'data/NIFTY50_all.csv')
        if not stock_path:
            return None, meta
        df = pd.read_csv(stock_path, parse_dates=['Date'])
    df = df[df['Series'] == 'EQ'].copy()
    valid_symbols = meta['Symbol'].tolist()
    df = df[df['Symbol'].isin(valid_symbols)].copy()
    df = df[['Date','Symbol','Open','High','Low','Close','Volume','Turnover','Prev Close','VWAP']].copy()
    df = df.sort_values(['Symbol','Date']).reset_index(drop=True)
    df = df[~((df['Open']==0) & (df['Close']==0))].copy()
    df = df.drop_duplicates(subset=['Date','Symbol']).copy()
    df = df.merge(meta[['Symbol','Industry','Company Name']], on='Symbol', how='left')
    df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
    df['Log_Return']   = df.groupby('Symbol')['Close'].transform(lambda x: np.log(x/x.shift(1)))
    df = df[df['Daily_Return'].abs() < 0.5].copy()
    return df, meta

@st.cache_data
def load_index_data():
    try:
        nifty_path = find_file(
            'Datasets/INDEX/NIFTY 50.csv',
            'data/NIFTY 50.csv',
            'nse_d2/Datasets/INDEX/NIFTY 50.csv',
        )
        nifty = pd.read_csv(nifty_path, parse_dates=['Date'])
        nifty = nifty[nifty['Date'] >= '2000-01-01'][['Date','Close']].dropna()
        nifty.columns = ['Date','NIFTY_Close']
    except:
        nifty = None
    try:
        vix_path = find_file(
            'Datasets/INDEX/INDIA VIX.csv',
            'data/INDIA VIX.csv',
            'nse_d2/Datasets/INDEX/INDIA VIX.csv',
        )
        vix = pd.read_csv(vix_path, parse_dates=['Date'])
        vix = vix[['Date','Close']].dropna()
        vix.columns = ['Date','VIX_Close']
    except:
        vix = None
    return nifty, vix

@st.cache_data
def compute_technicals(symbol_df):
    df = symbol_df.copy().sort_values('Date')
    close = df['Close']
    # Moving averages
    df['MA20']  = close.rolling(20).mean()
    df['MA50']  = close.rolling(50).mean()
    df['MA200'] = close.rolling(200).mean()
    df['EMA20'] = close.ewm(span=20, adjust=False).mean()
    # Bollinger Bands
    df['BB_Mid']   = close.rolling(20).mean()
    df['BB_Std']   = close.rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    # ATR
    df['TR']  = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR14'] = df['TR'].rolling(14).mean()
    # Volume MA
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    return df

@st.cache_data
def compute_portfolio_metrics(returns_df):
    """Compute annualized return, volatility, Sharpe, Sortino, Max Drawdown"""
    ann_return  = returns_df.mean() * 252
    ann_vol     = returns_df.std() * np.sqrt(252)
    sharpe      = ann_return / ann_vol
    downside    = returns_df[returns_df < 0].std() * np.sqrt(252)
    sortino     = ann_return / downside.replace(0, np.nan)
    cum         = (1 + returns_df).cumprod()
    roll_max    = cum.cummax()
    drawdown    = (cum - roll_max) / roll_max
    max_dd      = drawdown.min()
    return pd.DataFrame({
        'Ann_Return':   ann_return,
        'Ann_Vol':      ann_vol,
        'Sharpe':       sharpe,
        'Sortino':      sortino,
        'Max_Drawdown': max_dd
    })

@st.cache_data
def optimize_portfolio(returns_df, profile='balanced', n_sim=3000):
    """Monte Carlo portfolio optimization"""
    n  = returns_df.shape[1]
    mu = returns_df.mean() * 252
    cov = returns_df.cov() * 252
    syms = returns_df.columns.tolist()

    results = {'ret':[], 'vol':[], 'sharpe':[], 'weights':[]}
    np.random.seed(42)

    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n))
        r = w @ mu.values
        v = np.sqrt(w @ cov.values @ w)
        s = r / v if v > 0 else 0
        results['ret'].append(r)
        results['vol'].append(v)
        results['sharpe'].append(s)
        results['weights'].append(w)

    res_df = pd.DataFrame({'Return': results['ret'],
                           'Volatility': results['vol'],
                           'Sharpe': results['sharpe']})

    if profile == 'conservative':
        idx = res_df['Volatility'].idxmin()
    elif profile == 'aggressive':
        idx = res_df['Return'].idxmax()
    elif profile == 'balanced':
        idx = res_df['Sharpe'].idxmax()
    else:  # risk parity approx
        idx = res_df['Sharpe'].idxmax()

    best_w = results['weights'][idx]
    weights = pd.Series(best_w, index=syms)
    weights = weights[weights > 0.01].sort_values(ascending=False)
    weights = weights / weights.sum()

    return weights, res_df, results['weights']

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
# Check if NIFTY50_all.csv exists locally, else ask user to upload
_stock_path = find_file('NIFTY50_all.csv', 'data/NIFTY50_all.csv')
if _stock_path is None:
    st.sidebar.markdown("## 📂 Upload Required")
    uploaded_csv = st.sidebar.file_uploader(
        "Upload NIFTY50_all.csv to begin",
        type=['csv'],
        help="File is too large for GitHub. Upload it here once."
    )
    if uploaded_csv is None:
        st.title("📈 NIFTY-50 Investment Intelligence")
        st.info("👈 Please upload **NIFTY50_all.csv** in the sidebar to start. You can find it inside archive(3).zip.")
        st.stop()
else:
    uploaded_csv = None

with st.spinner('Loading market data...'):
    try:
        df, meta = load_data(uploaded_csv)
        if df is None:
            st.error("❌ Could not load stock data.")
            st.stop()
        nifty_index, vix = load_index_data()
        DATA_LOADED = True
    except Exception as e:
        st.error(f"Could not load data: {e}")
        DATA_LOADED = False
        st.stop()

symbols   = sorted(df['Symbol'].unique().tolist())
sectors   = sorted(df['Industry'].dropna().unique().tolist())
returns_wide = df.pivot_table(index='Date', columns='Symbol', values='Daily_Return')
returns_wide = returns_wide.dropna(thresh=500, axis=1)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 NIFTY-50 Intelligence")
    st.markdown("*IIT Roorkee Finance Club*")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠 Market Overview",
        "🔍 Stock Analysis",
        "💼 Portfolio Builder",
        "⚠️ Risk Assessment",
        "🚨 Anomaly Detection",
        "📊 Sector Analysis"
    ])

    st.markdown("---")
    st.markdown(f"**Data:** Jan 2000 – Apr 2021")
    st.markdown(f"**Stocks:** {df['Symbol'].nunique()}")
    st.markdown(f"**Records:** {len(df):,}")

# ─────────────────────────────────────────────
# PAGE 1: MARKET OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Market Overview":
    st.title("📈 NIFTY-50 Market Overview")
    st.caption("2000–2021 | National Stock Exchange of India")

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    latest_date = df['Date'].max()
    prev_date   = df[df['Date'] < latest_date]['Date'].max()

    mkt_return_today = df[df['Date'] == latest_date]['Daily_Return'].mean()
    mkt_return_prev  = df[df['Date'] == prev_date]['Daily_Return'].mean()

    with col1:
        st.metric("Market Avg Return (Latest Day)",
                  f"{mkt_return_today*100:.2f}%",
                  f"{(mkt_return_today - mkt_return_prev)*100:.2f}%")
    with col2:
        ann_vol = df['Daily_Return'].std() * np.sqrt(252) * 100
        st.metric("Market Annualized Volatility", f"{ann_vol:.1f}%")
    with col3:
        avg_sharpe = (df.groupby('Symbol')['Daily_Return'].mean() * 252 /
                      (df.groupby('Symbol')['Daily_Return'].std() * np.sqrt(252))).mean()
        st.metric("Avg Stock Sharpe Ratio", f"{avg_sharpe:.2f}")
    with col4:
        if vix is not None:
            latest_vix = vix['VIX_Close'].iloc[-1]
            st.metric("India VIX (Latest)", f"{latest_vix:.1f}",
                      delta_color="inverse")

    st.markdown("---")

    # NIFTY 50 Index Chart
    st.markdown('<p class="section-header">NIFTY-50 Index: 21-Year Journey</p>', unsafe_allow_html=True)

    if nifty_index is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nifty_index['Date'], y=nifty_index['NIFTY_Close'],
            fill='tozeroy', fillcolor='rgba(33,150,243,0.1)',
            line=dict(color='#2196F3', width=2),
            name='NIFTY 50'
        ))

        events = {
            '2000-03-10': 'Dot-com Bust',
            '2008-09-15': '2008 Crisis',
            '2020-03-23': 'COVID Crash',
            '2014-05-16': 'Modi Election',
        }
        colors_ev = {'Dot-com Bust':'red','2008 Crisis':'red',
                     'COVID Crash':'red','Modi Election':'green'}
        for date_str, label in events.items():
            d = pd.Timestamp(date_str)
            if d >= nifty_index['Date'].min():
                fig.add_vline(x=d, line_dash='dash',
                              line_color=colors_ev[label], opacity=0.6)
                fig.add_annotation(x=d, y=nifty_index['NIFTY_Close'].max()*0.85,
                                   text=label, showarrow=False,
                                   textangle=-90, font=dict(size=10, color=colors_ev[label]))

        fig.update_layout(
            template='plotly_dark', height=400,
            xaxis_title='Year', yaxis_title='Index Value',
            showlegend=False, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Market-wide returns distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Daily Returns Distribution</p>', unsafe_allow_html=True)
        fig2 = px.histogram(
            df['Daily_Return'].dropna().sample(10000, random_state=42),
            nbins=100, template='plotly_dark',
            color_discrete_sequence=['#2196F3'],
            labels={'value':'Daily Return'}
        )
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Top 10 All-Time Performers</p>', unsafe_allow_html=True)
        def total_ret(g, include_groups=False):
            g = g.sort_values('Date')
            if len(g) < 252: return np.nan
            return (g['Close'].iloc[-1]/g['Close'].iloc[0]-1)*100
        tr = df.groupby('Symbol', group_keys=False).apply(total_ret).dropna().sort_values(ascending=False).head(10)
        fig3 = px.bar(
            x=tr.values, y=tr.index,
            orientation='h', template='plotly_dark',
            color=tr.values, color_continuous_scale='Greens',
            labels={'x':'Total Return (%)','y':'Symbol'}
        )
        fig3.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    # VIX
    if vix is not None:
        st.markdown('<p class="section-header">India VIX — Volatility & Fear Index</p>', unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=vix['Date'], y=vix['VIX_Close'],
            fill='tozeroy', fillcolor='rgba(244,67,54,0.15)',
            line=dict(color='#f44336', width=1.5), name='VIX'
        ))
        fig4.add_hline(y=20, line_dash='dash', line_color='orange',
                       annotation_text='Moderate Volatility (20)')
        fig4.add_hline(y=30, line_dash='dash', line_color='red',
                       annotation_text='High Volatility (30)')
        fig4.update_layout(template='plotly_dark', height=300,
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2: STOCK ANALYSIS
# ─────────────────────────────────────────────
elif page == "🔍 Stock Analysis":
    st.title("🔍 Individual Stock Analysis")

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        symbol = st.selectbox("Select Stock", symbols, index=symbols.index('RELIANCE') if 'RELIANCE' in symbols else 0)
    with col2:
        date_from = st.date_input("From", value=pd.Timestamp('2015-01-01'))
    with col3:
        date_to   = st.date_input("To",   value=pd.Timestamp('2021-04-30'))

    stock_df = df[df['Symbol'] == symbol].copy()
    stock_df = stock_df[(stock_df['Date'] >= pd.Timestamp(date_from)) &
                        (stock_df['Date'] <= pd.Timestamp(date_to))]
    stock_df = compute_technicals(stock_df)

    company_name = meta[meta['Symbol']==symbol]['Company Name'].values
    company_name = company_name[0] if len(company_name) > 0 else symbol
    industry     = meta[meta['Symbol']==symbol]['Industry'].values
    industry     = industry[0] if len(industry) > 0 else 'N/A'

    st.markdown(f"### {company_name} ({symbol}) — {industry}")

    # KPI row
    if len(stock_df) > 1:
        latest_close = stock_df['Close'].iloc[-1]
        prev_close   = stock_df['Close'].iloc[-2]
        change_pct   = (latest_close - prev_close) / prev_close * 100
        total_ret_v  = (stock_df['Close'].iloc[-1]/stock_df['Close'].iloc[0]-1)*100
        vol_ann      = stock_df['Daily_Return'].std() * np.sqrt(252) * 100
        sharpe_v     = (stock_df['Daily_Return'].mean()*252) / (stock_df['Daily_Return'].std()*np.sqrt(252))
        latest_rsi   = stock_df['RSI'].iloc[-1] if 'RSI' in stock_df.columns else np.nan

        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Latest Close",  f"₹{latest_close:,.2f}", f"{change_pct:+.2f}%")
        col2.metric("Total Return",  f"{total_ret_v:+.1f}%")
        col3.metric("Ann. Volatility", f"{vol_ann:.1f}%")
        col4.metric("Sharpe Ratio",  f"{sharpe_v:.2f}")
        col5.metric("RSI(14)",       f"{latest_rsi:.1f}" if not np.isnan(latest_rsi) else "N/A")

    st.markdown("---")

    # Candlestick + Volume
    st.markdown('<p class="section-header">Price Chart with Technical Indicators</p>', unsafe_allow_html=True)

    show_ma      = st.checkbox("Moving Averages (MA20, MA50, MA200)", value=True)
    show_bb      = st.checkbox("Bollinger Bands", value=False)
    show_signals = st.checkbox("Buy/Sell Signals (MA Crossover)", value=False)

    plot_df = stock_df.tail(500)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.03,
                        subplot_titles=['Price', 'Volume', 'RSI(14)'])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'],
        increasing_line_color='#4CAF50', decreasing_line_color='#f44336',
        name='OHLC'
    ), row=1, col=1)

    if show_ma:
        for ma, color in [('MA20','#FFC107'),('MA50','#2196F3'),('MA200','#9C27B0')]:
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df[ma],
                                     line=dict(color=color, width=1.2),
                                     name=ma), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_Upper'],
                                 line=dict(color='gray', width=1, dash='dash'),
                                 name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_Lower'],
                                 fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                                 line=dict(color='gray', width=1, dash='dash'),
                                 name='BB Lower'), row=1, col=1)

    if show_signals and 'MA20' in plot_df.columns and 'MA50' in plot_df.columns:
        buy_signals  = plot_df[(plot_df['MA20'] > plot_df['MA50']) &
                               (plot_df['MA20'].shift(1) <= plot_df['MA50'].shift(1))]
        sell_signals = plot_df[(plot_df['MA20'] < plot_df['MA50']) &
                               (plot_df['MA20'].shift(1) >= plot_df['MA50'].shift(1))]
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Low']*0.98,
                                 mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'),
                                 name='Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['High']*1.02,
                                 mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                                 name='Sell Signal'), row=1, col=1)

    # Volume
    colors_vol = ['#4CAF50' if r >= 0 else '#f44336'
                  for r in plot_df['Daily_Return'].fillna(0)]
    fig.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Volume'],
                         marker_color=colors_vol, name='Volume',
                         opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Vol_MA20'],
                             line=dict(color='white', width=1),
                             name='Vol MA20'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['RSI'],
                             line=dict(color='#FF9800', width=1.5),
                             name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red',   row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='red',   opacity=0.05, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='green', opacity=0.05, row=3, col=1)

    fig.update_layout(template='plotly_dark', height=700,
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # MACD
    st.markdown('<p class="section-header">MACD</p>', unsafe_allow_html=True)
    fig_macd = make_subplots(rows=1, cols=1)
    fig_macd.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MACD'],
                                   line=dict(color='#2196F3'), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MACD_Signal'],
                                   line=dict(color='#FF9800', dash='dash'), name='Signal'))
    colors_macd = ['#4CAF50' if v >= 0 else '#f44336' for v in plot_df['MACD_Hist'].fillna(0)]
    fig_macd.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['MACD_Hist'],
                               marker_color=colors_macd, name='Histogram', opacity=0.6))
    fig_macd.update_layout(template='plotly_dark', height=250,
                            margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_macd, use_container_width=True)

    # Stats table
    st.markdown('<p class="section-header">Descriptive Statistics</p>', unsafe_allow_html=True)
    stats = stock_df[['Open','High','Low','Close','Volume']].describe().round(2)
    st.dataframe(stats, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3: PORTFOLIO BUILDER
# ─────────────────────────────────────────────
elif page == "💼 Portfolio Builder":
    st.title("💼 Smart Portfolio Builder")
    st.caption("Monte Carlo optimization across 3,000 simulated portfolios")

    col1, col2 = st.columns([1, 2])
    with col1:
        profile = st.selectbox("Investor Profile", [
            "Conservative (Min Volatility)",
            "Balanced (Max Sharpe)",
            "Aggressive (Max Return)"
        ])
        profile_key = {'Conservative (Min Volatility)':'conservative',
                       'Balanced (Max Sharpe)':'balanced',
                       'Aggressive (Max Return)':'aggressive'}[profile]

        selected_sectors = st.multiselect(
            "Filter by Sector (optional)",
            options=sectors, default=[]
        )

        top_n = st.slider("Max number of stocks in portfolio", 5, 20, 10)

        start_year = st.selectbox("Backtest Start Year", [2010, 2012, 2015, 2018], index=0)

    # Filter symbols by sector
    if selected_sectors:
        filtered_meta = meta[meta['Industry'].isin(selected_sectors)]
        filtered_syms = filtered_meta['Symbol'].tolist()
        ret_df = returns_wide[[c for c in returns_wide.columns if c in filtered_syms]]
    else:
        ret_df = returns_wide.copy()

    ret_df = ret_df[ret_df.index >= pd.Timestamp(f'{start_year}-01-01')]
    ret_df = ret_df.dropna(thresh=int(len(ret_df)*0.8), axis=1)
    ret_df = ret_df.fillna(0)

    if len(ret_df.columns) < 3:
        st.warning("Not enough stocks after filtering. Please remove sector filters.")
        st.stop()

    with st.spinner("Optimizing portfolio across 3,000 simulations..."):
        weights, sim_df, all_weights = optimize_portfolio(ret_df, profile=profile_key)

    # Keep top N
    weights = weights.head(top_n)
    weights = weights / weights.sum()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<p class="section-header">Optimal Allocation</p>', unsafe_allow_html=True)
        w_df = pd.DataFrame({'Symbol': weights.index, 'Weight': weights.values})
        w_df['Weight_%'] = (w_df['Weight']*100).round(2)
        w_df = w_df.merge(meta[['Symbol','Company Name','Industry']], on='Symbol', how='left')

        fig_pie = px.pie(w_df, values='Weight_%', names='Symbol',
                         hover_data=['Company Name','Industry'],
                         template='plotly_dark',
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Efficient Frontier</p>', unsafe_allow_html=True)
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=sim_df['Volatility']*100, y=sim_df['Return']*100,
            mode='markers',
            marker=dict(color=sim_df['Sharpe'], colorscale='Viridis',
                        size=3, opacity=0.5,
                        colorbar=dict(title='Sharpe')),
            name='Simulated Portfolios'
        ))
        # Mark selected portfolio
        port_ret = (ret_df[weights.index] * weights.values).sum(axis=1)
        p_ret = port_ret.mean() * 252 * 100
        p_vol = port_ret.std() * np.sqrt(252) * 100
        fig_ef.add_trace(go.Scatter(
            x=[p_vol], y=[p_ret],
            mode='markers',
            marker=dict(symbol='star', size=18, color='red'),
            name='Selected Portfolio'
        ))
        fig_ef.update_layout(template='plotly_dark', height=400,
                              xaxis_title='Volatility (%)',
                              yaxis_title='Expected Return (%)',
                              margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_ef, use_container_width=True)

    # Portfolio metrics
    st.markdown('<p class="section-header">Portfolio Performance Metrics</p>', unsafe_allow_html=True)

    port_returns = (ret_df[weights.index] * weights.values).sum(axis=1)
    cum_ret      = (1 + port_returns).cumprod()
    ann_ret      = port_returns.mean() * 252 * 100
    ann_vol_p    = port_returns.std() * np.sqrt(252) * 100
    sharpe_p     = (port_returns.mean()*252) / (port_returns.std()*np.sqrt(252))
    downside_p   = port_returns[port_returns<0].std() * np.sqrt(252)
    sortino_p    = (port_returns.mean()*252) / downside_p if downside_p > 0 else np.nan
    roll_max_p   = cum_ret.cummax()
    max_dd_p     = ((cum_ret - roll_max_p)/roll_max_p).min() * 100

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Ann. Return",    f"{ann_ret:.1f}%")
    c2.metric("Ann. Volatility",f"{ann_vol_p:.1f}%")
    c3.metric("Sharpe Ratio",   f"{sharpe_p:.2f}")
    c4.metric("Sortino Ratio",  f"{sortino_p:.2f}" if not np.isnan(sortino_p) else "N/A")
    c5.metric("Max Drawdown",   f"{max_dd_p:.1f}%")

    # Cumulative return chart
    st.markdown('<p class="section-header">Portfolio Cumulative Return</p>', unsafe_allow_html=True)
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=cum_ret.index, y=cum_ret.values,
        fill='tozeroy', fillcolor='rgba(76,175,80,0.1)',
        line=dict(color='#4CAF50', width=2),
        name='Portfolio'
    ))
    if nifty_index is not None:
        nifty_filtered = nifty_index[nifty_index['Date'] >= cum_ret.index.min()].copy()
        nifty_norm = nifty_filtered['NIFTY_Close'] / nifty_filtered['NIFTY_Close'].iloc[0]
        fig_cum.add_trace(go.Scatter(
            x=nifty_filtered['Date'], y=nifty_norm.values,
            line=dict(color='#2196F3', width=1.5, dash='dash'),
            name='NIFTY 50 Benchmark'
        ))
    fig_cum.update_layout(template='plotly_dark', height=350,
                           yaxis_title='Growth of ₹1',
                           margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_cum, use_container_width=True)

    # Allocation table
    st.markdown('<p class="section-header">Allocation Details</p>', unsafe_allow_html=True)
    display_df = w_df[['Symbol','Company Name','Industry','Weight_%']].copy()
    display_df.columns = ['Symbol','Company','Sector','Weight (%)']
    st.dataframe(display_df.sort_values('Weight (%)', ascending=False),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 4: RISK ASSESSMENT
# ─────────────────────────────────────────────
elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment Dashboard")

    tab1, tab2 = st.tabs(["📊 Per-Stock Risk", "📉 Drawdown Analysis"])

    with tab1:
        st.markdown('<p class="section-header">Risk Metrics — All Stocks</p>', unsafe_allow_html=True)

        metrics = compute_portfolio_metrics(returns_wide)
        metrics = metrics.reset_index()
        metrics.columns = ['Symbol','Ann_Return','Ann_Vol','Sharpe','Sortino','Max_Drawdown']
        metrics = metrics.merge(meta[['Symbol','Industry']], on='Symbol', how='left')
        metrics['Ann_Return']   = (metrics['Ann_Return']*100).round(2)
        metrics['Ann_Vol']      = (metrics['Ann_Vol']*100).round(2)
        metrics['Sharpe']       = metrics['Sharpe'].round(3)
        metrics['Sortino']      = metrics['Sortino'].round(3)
        metrics['Max_Drawdown'] = (metrics['Max_Drawdown']*100).round(2)

        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Sort by", ['Sharpe','Ann_Return','Ann_Vol','Max_Drawdown'])
        with col2:
            sector_filter = st.selectbox("Filter Sector", ['All'] + sectors)

        display = metrics.copy()
        if sector_filter != 'All':
            display = display[display['Industry'] == sector_filter]
        display = display.sort_values(sort_by, ascending=(sort_by in ['Ann_Vol','Max_Drawdown']))

        # Color-coded table
        def color_sharpe(val):
            if val > 1:   return 'background-color: #1b5e20; color: white'
            elif val > 0: return 'background-color: #388e3c; color: white'
            else:         return 'background-color: #b71c1c; color: white'
        def color_drawdown(val):
            if val < -50: return 'background-color: #b71c1c; color: white'
            elif val < -30: return 'background-color: #e65100; color: white'
            else:         return 'background-color: #1b5e20; color: white'

        st.dataframe(
            display[['Symbol','Industry','Ann_Return','Ann_Vol','Sharpe','Sortino','Max_Drawdown']].rename(columns={
                'Ann_Return':'Return %','Ann_Vol':'Volatility %',
                'Max_Drawdown':'Max DD %'
            }),
            use_container_width=True, hide_index=True, height=400
        )

        # Risk-return scatter
        st.markdown('<p class="section-header">Risk-Return Scatter</p>', unsafe_allow_html=True)
        metrics['Sharpe_size'] = metrics['Sharpe'].clip(lower=0.1)
        fig_rr = px.scatter(
            metrics, x='Ann_Vol', y='Ann_Return',
            color='Industry', size='Sharpe_size',
            size_max=20, hover_data=['Symbol','Sharpe','Max_Drawdown'],
            template='plotly_dark', labels={
                'Ann_Vol':'Annualized Volatility (%)',
                'Ann_Return':'Annualized Return (%)'
            }
        )
        fig_rr.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
        fig_rr.update_layout(height=450, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_rr, use_container_width=True)

    with tab2:
        st.markdown('<p class="section-header">Drawdown Analysis</p>', unsafe_allow_html=True)
        sym_dd = st.selectbox("Select Stock for Drawdown", symbols)

        dd_df = df[df['Symbol'] == sym_dd].sort_values('Date').copy()
        dd_df['Cum_Return']  = (1 + dd_df['Daily_Return'].fillna(0)).cumprod()
        dd_df['Rolling_Max'] = dd_df['Cum_Return'].cummax()
        dd_df['Drawdown']    = (dd_df['Cum_Return'] - dd_df['Rolling_Max']) / dd_df['Rolling_Max'] * 100

        fig_dd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_dd.add_trace(go.Scatter(x=dd_df['Date'], y=dd_df['Close'],
                                    line=dict(color='#2196F3'), name='Price'), row=1, col=1)
        fig_dd.add_trace(go.Scatter(x=dd_df['Date'], y=dd_df['Drawdown'],
                                    fill='tozeroy', fillcolor='rgba(244,67,54,0.2)',
                                    line=dict(color='#f44336'), name='Drawdown %'), row=2, col=1)
        fig_dd.update_layout(template='plotly_dark', height=500,
                             margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_dd, use_container_width=True)

        max_dd_val = dd_df['Drawdown'].min()
        max_dd_date = dd_df.loc[dd_df['Drawdown'].idxmin(), 'Date']
        st.metric("Maximum Drawdown", f"{max_dd_val:.1f}%", f"on {max_dd_date.date()}")


# ─────────────────────────────────────────────
# PAGE 5: ANOMALY DETECTION
# ─────────────────────────────────────────────
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Market Anomaly Detection")
    st.caption("Detecting unusual market events using statistical methods")

    tab1, tab2 = st.tabs(["📈 Market-Wide Anomalies", "🔍 Stock-Level Anomalies"])

    with tab1:
        st.markdown('<p class="section-header">Market Crash & Spike Detection</p>', unsafe_allow_html=True)

        # Rolling z-score on market average return
        market_daily = df.groupby('Date')['Daily_Return'].mean().reset_index()
        market_daily.columns = ['Date','Market_Return']
        market_daily = market_daily.sort_values('Date')
        market_daily['Rolling_Mean'] = market_daily['Market_Return'].rolling(60).mean()
        market_daily['Rolling_Std']  = market_daily['Market_Return'].rolling(60).std()
        market_daily['Z_Score'] = ((market_daily['Market_Return'] - market_daily['Rolling_Mean']) /
                                    market_daily['Rolling_Std'])

        threshold = st.slider("Z-Score Threshold for Anomaly", 1.5, 4.0, 2.5, 0.1)
        anomalies = market_daily[market_daily['Z_Score'].abs() > threshold]

        fig_anom = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.5, 0.5], vertical_spacing=0.05)

        fig_anom.add_trace(go.Scatter(x=market_daily['Date'], y=market_daily['Market_Return']*100,
                                      line=dict(color='#2196F3', width=1), name='Market Return %'), row=1, col=1)
        fig_anom.add_trace(go.Scatter(
            x=anomalies['Date'], y=anomalies['Market_Return']*100,
            mode='markers', marker=dict(color='red', size=8, symbol='x'),
            name='Anomaly'), row=1, col=1)

        fig_anom.add_trace(go.Scatter(x=market_daily['Date'], y=market_daily['Z_Score'],
                                      line=dict(color='#FF9800', width=1), name='Z-Score'), row=2, col=1)
        fig_anom.add_hline(y=threshold,  line_dash='dash', line_color='red',   row=2, col=1)
        fig_anom.add_hline(y=-threshold, line_dash='dash', line_color='green', row=2, col=1)

        fig_anom.update_layout(template='plotly_dark', height=500,
                               margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown(f"**{len(anomalies)} anomaly days detected** (|Z| > {threshold})")

        # Top anomaly events
        top_anom = anomalies.nlargest(5, 'Z_Score')[['Date','Market_Return','Z_Score']]
        bot_anom = anomalies.nsmallest(5, 'Z_Score')[['Date','Market_Return','Z_Score']]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 Positive Shocks**")
            top_anom['Market_Return'] = (top_anom['Market_Return']*100).round(2)
            st.dataframe(top_anom.reset_index(drop=True), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Top 5 Negative Shocks (Crashes)**")
            bot_anom['Market_Return'] = (bot_anom['Market_Return']*100).round(2)
            st.dataframe(bot_anom.reset_index(drop=True), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown('<p class="section-header">Stock-Level Volume & Return Anomalies</p>', unsafe_allow_html=True)

        sym_an = st.selectbox("Select Stock", symbols, key='anomaly_stock')
        s_df = df[df['Symbol'] == sym_an].sort_values('Date').copy()

        s_df['Return_Z']  = (s_df['Daily_Return'] - s_df['Daily_Return'].rolling(60).mean()) / \
                             s_df['Daily_Return'].rolling(60).std()
        s_df['Volume_Z']  = (s_df['Volume'] - s_df['Volume'].rolling(60).mean()) / \
                             s_df['Volume'].rolling(60).std()
        s_df['Anomaly']   = (s_df['Return_Z'].abs() > 2.5) | (s_df['Volume_Z'].abs() > 2.5)

        fig_s = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               row_heights=[0.4,0.3,0.3], vertical_spacing=0.04)
        fig_s.add_trace(go.Scatter(x=s_df['Date'], y=s_df['Close'],
                                   line=dict(color='#2196F3'), name='Price'), row=1, col=1)
        anom_s = s_df[s_df['Anomaly']]
        fig_s.add_trace(go.Scatter(x=anom_s['Date'], y=anom_s['Close'],
                                   mode='markers', marker=dict(color='red', size=7, symbol='x'),
                                   name='Anomaly'), row=1, col=1)
        fig_s.add_trace(go.Bar(x=s_df['Date'], y=s_df['Volume'],
                               marker_color='#FF9800', opacity=0.5, name='Volume'), row=2, col=1)
        fig_s.add_trace(go.Scatter(x=s_df['Date'], y=s_df['Return_Z'],
                                   line=dict(color='#9C27B0', width=1), name='Return Z-Score'), row=3, col=1)
        fig_s.add_hline(y=2.5,  line_dash='dash', line_color='red',   row=3, col=1)
        fig_s.add_hline(y=-2.5, line_dash='dash', line_color='green', row=3, col=1)

        fig_s.update_layout(template='plotly_dark', height=550,
                            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_s, use_container_width=True)
        st.info(f"**{anom_s.shape[0]} anomaly days** detected for {sym_an}")


# ─────────────────────────────────────────────
# PAGE 6: SECTOR ANALYSIS
# ─────────────────────────────────────────────
elif page == "📊 Sector Analysis":
    st.title("📊 Sector Deep Dive")

    selected_sector = st.selectbox("Select Sector", sectors)
    sector_stocks = meta[meta['Industry'] == selected_sector]['Symbol'].tolist()
    sector_df = df[df['Symbol'].isin(sector_stocks)].copy()

    st.markdown(f"**{len(sector_stocks)} stocks** in {selected_sector}")
    st.markdown(", ".join(sector_stocks))
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        # Sector cumulative return
        st.markdown('<p class="section-header">Cumulative Returns by Stock</p>', unsafe_allow_html=True)
        fig_sec = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, sym in enumerate(sector_stocks):
            s = df[df['Symbol']==sym].sort_values('Date')
            if len(s) < 100: continue
            cum = (1 + s['Daily_Return'].fillna(0)).cumprod()
            fig_sec.add_trace(go.Scatter(x=s['Date'], y=cum.values,
                                         name=sym,
                                         line=dict(color=colors[i % len(colors)], width=1.5)))
        fig_sec.update_layout(template='plotly_dark', height=350,
                               yaxis_title='Cumulative Return',
                               margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_sec, use_container_width=True)

    with col2:
        # Risk-return within sector
        st.markdown('<p class="section-header">Risk-Return Profile</p>', unsafe_allow_html=True)
        sec_metrics = []
        for sym in sector_stocks:
            s = df[df['Symbol']==sym]['Daily_Return'].dropna()
            if len(s) < 100: continue
            sec_metrics.append({
                'Symbol': sym,
                'Return': s.mean()*252*100,
                'Volatility': s.std()*np.sqrt(252)*100,
                'Sharpe': (s.mean()*252)/(s.std()*np.sqrt(252)) if s.std() > 0 else 0
            })
        sec_m_df = pd.DataFrame(sec_metrics)
        fig_sm = px.scatter(sec_m_df, x='Volatility', y='Return',
                            text='Symbol', color='Sharpe',
                            color_continuous_scale='RdYlGn',
                            template='plotly_dark',
                            labels={'Return':'Ann. Return (%)','Volatility':'Ann. Vol (%)'},
                            size_max=15)
        fig_sm.update_traces(textposition='top center')
        fig_sm.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_sm, use_container_width=True)

    # Sector correlation heatmap
    st.markdown('<p class="section-header">Intra-Sector Correlation</p>', unsafe_allow_html=True)
    sec_ret = returns_wide[[c for c in returns_wide.columns if c in sector_stocks]]
    if sec_ret.shape[1] > 1:
        corr = sec_ret.corr()
        fig_corr = px.imshow(corr, template='plotly_dark',
                             color_continuous_scale='RdYlGn',
                             zmin=-1, zmax=1,
                             text_auto='.2f')
        fig_corr.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need at least 2 stocks for correlation analysis.")

    # Monthly seasonality
    st.markdown('<p class="section-header">Monthly Return Seasonality</p>', unsafe_allow_html=True)
    sector_df['Month'] = sector_df['Date'].dt.month
    monthly_ret = sector_df.groupby('Month')['Daily_Return'].mean() * 21 * 100
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_ret.index = [month_names[i-1] for i in monthly_ret.index]
    colors_m = ['#4CAF50' if v >= 0 else '#f44336' for v in monthly_ret.values]
    fig_month = go.Figure(go.Bar(
        x=monthly_ret.index, y=monthly_ret.values,
        marker_color=colors_m, text=[f'{v:.1f}%' for v in monthly_ret.values],
        textposition='outside'
    ))
    fig_month.update_layout(template='plotly_dark', height=320,
                             yaxis_title='Avg Monthly Return (%)',
                             margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_month, use_container_width=True)
