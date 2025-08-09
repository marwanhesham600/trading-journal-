"""
Advanced Trading Journal - Streamlit Prototype (Password Protected)

Features:
- Password protection using Streamlit Secrets (secret key: JOURNAL_PASSWORD)
- CSV/Excel import
- SQLite backend (local file: trades.db)
- Interactive filters, charts (plotly)
- Export filtered CSV

How to run locally:
1. create a virtualenv and install: pip install streamlit pandas plotly sqlalchemy openpyxl
2. create a secrets.toml with your password for local testing:
   [secrets]
   JOURNAL_PASSWORD = "<PASTE_YOUR_PASSWORD_HERE>"
3. run: streamlit run Advanced_Trading_Journal_Streamlit.py

Deployment notes:
- On Streamlit Community Cloud, set a secret with the key JOURNAL_PASSWORD and the password value.
- The app will read st.secrets['JOURNAL_PASSWORD'] to authenticate users.

CSV template columns (header row):
entry_dt,exit_dt,symbol,side,size,entry_price,exit_price,pnl,percent_risk,rr,tags,context,session,notes,emotion,trade_url
- datetime format: YYYY-MM-DD HH:MM:SS (24h)
- tags: semicolon-separated (e.g. FVG;SMT;Liquidity)

This file is intended as a starting prototype. You can later add automatic imports (MT5/TradeLocker), auth, screenshots upload, and cloud DB.
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import os

DB_PATH = "trades.db"
TABLE = "trades"

# ---------- Simple password-protection ----------
# The app expects a secret named JOURNAL_PASSWORD in st.secrets.
# On Streamlit Cloud: set this secret in the app settings (Secrets).
# Locally: create a file .streamlit/secrets.toml with content:
# [secrets]
# JOURNAL_PASSWORD = "your_password_here"

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Show login form if not authenticated
if not st.session_state['authenticated']:
    st.title("Trading Journal — Login")
    pwd = st.text_input("Enter password", type='password')
    if st.button("Login"):
        try:
            secret_pwd = st.secrets.get('JOURNAL_PASSWORD')
            if secret_pwd is None:
                st.error("No password found in Streamlit secrets. For local testing, create .streamlit/secrets.toml with JOURNAL_PASSWORD.")
            elif pwd == secret_pwd:
                st.session_state['authenticated'] = True
                st.experimental_rerun()
            else:
                st.error("Wrong password.")
        except Exception as e:
            st.error(f"Error checking secret: {e}")
    st.stop()

# ---------- Database helpers ----------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_dt TEXT,
        exit_dt TEXT,
        symbol TEXT,
        side TEXT,
        size REAL,
        entry_price REAL,
        exit_price REAL,
        pnl REAL,
        percent_risk REAL,
        rr REAL,
        tags TEXT,
        context TEXT,
        session TEXT,
        notes TEXT,
        emotion TEXT,
        trade_url TEXT
    )
    """)
    conn.commit()
    conn.close()


def insert_trades_from_df(df: pd.DataFrame):
    expected = [
        "entry_dt","exit_dt","symbol","side","size","entry_price","exit_price",
        "pnl","percent_risk","rr","tags","context","session","notes","emotion","trade_url"
    ]
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            df[col] = None
    for col in ["entry_dt","exit_dt"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    conn = sqlite3.connect(DB_PATH)
    df[expected].to_sql(TABLE, conn, if_exists='append', index=False)
    conn.close()


def load_all_trades() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    return df

# ---------- Metrics & utilities ----------

def compute_metrics(df: pd.DataFrame) -> dict:
    out = {}
    if df.empty:
        return out
    df = df.copy()
    for col in ["pnl","rr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    out['total_trades'] = len(df)
    out['win_rate'] = len(wins) / len(df) if len(df)>0 else 0
    out['avg_rr'] = df['rr'].mean() if 'rr' in df.columns else None
    out['avg_pnl'] = df['pnl'].mean()
    out['expectancy'] = ( (len(wins)/len(df) * (wins['pnl'].mean() if len(wins)>0 else 0)) + ((len(losses)/len(df)) * (losses['pnl'].mean() if len(losses)>0 else 0)) ) if len(df)>0 else 0
    df_sorted = df.sort_values('exit_dt') if 'exit_dt' in df.columns else df
    df_sorted['cum_pnl'] = df_sorted['pnl'].cumsum()
    out['equity_curve'] = df_sorted[['exit_dt','cum_pnl']]
    out['win_streak'], out['lose_streak'] = longest_streaks(df['pnl'])
    out['by_symbol'] = df.groupby('symbol').agg(trades=('pnl','count'), pnl=('pnl','sum'), win_rate=('pnl', lambda x: (x>0).sum()/len(x))).reset_index()
    return out


def longest_streaks(pnl_series: pd.Series):
    max_win = max_loss = 0
    cur_win = cur_loss = 0
    for p in pnl_series.fillna(0):
        if p > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def parse_tags(tags_cell):
    if pd.isna(tags_cell):
        return []
    if isinstance(tags_cell, str):
        return [t.strip() for t in tags_cell.split(';') if t.strip()]
    if isinstance(tags_cell, (list,tuple)):
        return list(tags_cell)
    return []

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Advanced Trading Journal", layout='wide')
init_db()

st.title("Advanced Trading Journal — Private")

# Sidebar: Import and Filters
with st.sidebar:
    st.header("Import / Data")
    uploaded = st.file_uploader("Upload trades CSV or Excel (template)", type=['csv','xlsx'])
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            st.write(f"Preview: {len(df_up)} rows")
            st.dataframe(df_up.head())
            if st.button("Import these trades to DB"):
                insert_trades_from_df(df_up)
                st.success("Imported!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if st.button("Clear all trades (danger)"):
        if st.confirm("Are you sure? This will delete all trades in local DB."):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(f"DELETE FROM {TABLE}")
            conn.commit()
            conn.close()
            st.success("Deleted all trades")

    st.markdown("---")
    st.header("Filters")
    df_all = load_all_trades()
    if df_all.empty:
        st.info("No trades in DB yet — import a CSV/xlsx to get started.")
    else:
        for col in ['entry_dt','exit_dt']:
            if col in df_all.columns:
                try:
                    df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
                except:
                    pass
        symbols = sorted(df_all['symbol'].dropna().unique())
        sessions = sorted(df_all['session'].dropna().unique())
        tags_list = sorted({t for cell in df_all['tags'].dropna().unique() for t in parse_tags(cell)})

        sel_symbols = st.multiselect("Symbols", options=symbols, default=symbols)
        sel_sessions = st.multiselect("Sessions", options=sessions, default=sessions)
        sel_tags = st.multiselect("Tags (AND filter)", options=tags_list)
        date_min = df_all['exit_dt'].min() if 'exit_dt' in df_all.columns else None
        date_max = df_all['exit_dt'].max() if 'exit_dt' in df_all.columns else None
        date_range = st.date_input("Exit date range", value=(date_min.date() if date_min is not None else None, date_max.date() if date_max is not None else None))

# Main area: show stats and charts
if not df_all.empty:
    df = df_all.copy()
    if 'symbol' in df.columns and sel_symbols:
        df = df[df['symbol'].isin(sel_symbols)]
    if 'session' in df.columns and sel_sessions:
        df = df[df['session'].isin(sel_sessions)]
    if sel_tags:
        def has_all_tags(cell):
            parsed = parse_tags(cell)
            return all(t in parsed for t in sel_tags)
        df = df[df['tags'].apply(has_all_tags)]
    try:
        if date_range and len(date_range)==2:
            start = pd.to_datetime(date_range[0])
            end = pd.to_datetime(date_range[1])
            if 'exit_dt' in df.columns:
                df = df[(df['exit_dt'] >= start) & (df['exit_dt'] <= (end + pd.Timedelta(days=1)))]
    except Exception:
        pass

    st.subheader("Quick metrics")
    metrics = compute_metrics(df)
    if metrics:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Trades", metrics.get('total_trades',0))
        c2.metric("Win rate", f"{metrics.get('win_rate',0)*100:.1f}%")
        c3.metric("Avg PnL", f"{metrics.get('avg_pnl',0):.2f}")
        c4.metric("Expectancy", f"{metrics.get('expectancy',0):.2f}")

    st.subheader("Equity Curve")
    eq = metrics.get('equity_curve')
    if eq is not None and not eq.empty:
        eq = eq.copy()
        eq['exit_dt'] = pd.to_datetime(eq['exit_dt'], errors='coerce')
        fig = px.line(eq, x='exit_dt', y='cum_pnl', title='Cumulative PnL')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity data yet.")

    st.subheader("Daily PnL")
    if 'exit_dt' in df.columns:
        df['date'] = pd.to_datetime(df['exit_dt']).dt.date
        daily = df.groupby('date').agg(daily_pnl=('pnl','sum')).reset_index()
        fig2 = px.bar(daily, x='date', y='daily_pnl', title='Daily PnL')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Performance by Symbol")
    if 'by_symbol' in metrics:
        st.dataframe(metrics['by_symbol'])
        fig3 = px.bar(metrics['by_symbol'], x='symbol', y='pnl', title='PnL by symbol')
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("RR vs PnL")
    if 'rr' in df.columns and not df['rr'].isna().all():
        fig4 = px.scatter(df, x='rr', y='pnl', hover_data=['symbol','entry_dt','exit_dt','tags'])
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Tag analysis")
    all_tags = []
    for cell in df['tags'].dropna().unique():
        all_tags += parse_tags(cell)
    if all_tags:
        tag_series = pd.Series(all_tags).value_counts().reset_index()
        tag_series.columns = ['tag','count']
        st.dataframe(tag_series)
        fig5 = px.bar(tag_series, x='tag', y='count', title='Tag counts')
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Filtered trades")
    st.dataframe(df.sort_values('exit_dt', ascending=False).reset_index(drop=True))

    csv = df.to_csv(index=False)
    st.download_button("Download filtered CSV", csv, file_name='trades_filtered.csv', mime='text/csv')

    st.subheader("Trade review template")
    st.write("Select a trade by ID to open quick review fields.")
    trade_ids = df['id'].tolist()
    sel = st.selectbox("Select trade ID", options=trade_ids)
    if sel:
        tr = df[df['id']==sel].iloc[0]
        st.markdown(f"**Symbol:** {tr.get('symbol','')}, **Entry:** {tr.get('entry_price','')}, **Exit:** {tr.get('exit_price','')}")

**Notes:** {tr.get('notes','')}

**Tags:** {tr.get('tags','')}")
        review_plan = st.text_area("What was the plan / rule for this trade?", value="")
        review_result = st.text_area("What actually happened?", value="")
        review_lesson = st.text_area("Key lesson / action to take next time", value="")
        if st.button("Save review to notes"):
            new_notes = (str(tr.get('notes','')) + '

--- REVIEW ---
Plan: ' + review_plan + '
Result: ' + review_result + '
Lesson: ' + review_lesson)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(f"UPDATE {TABLE} SET notes = ? WHERE id = ?", (new_notes, sel))
            conn.commit()
            conn.close()
            st.success("Saved review into trade notes")

else:
    st.write("No data to show. Use the sidebar to import trades from a CSV or Excel file.")

st.markdown("---")
st.caption("Prototype by Zane. Follow the deployment guide in the chat to publish to Streamlit Cloud and set your password secret.")
