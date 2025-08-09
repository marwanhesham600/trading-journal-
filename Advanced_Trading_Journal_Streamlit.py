import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import os
from datetime import datetime

# ------------------
# PASSWORD PROTECTION
# ------------------

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["JOURNAL_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ------------------
# DATABASE SETUP
# ------------------

DB_FILE = "trading_journal.db"

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS trades (
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
            )''')
conn.commit()

# ------------------
# FILE UPLOAD
# ------------------

st.title("📈 Advanced Trading Journal")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.to_sql("trades", conn, if_exists="append", index=False)
        st.success(f"Uploaded {len(df)} trades successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ------------------
# LOAD DATA
# ------------------

df = pd.read_sql("SELECT * FROM trades", conn)

if df.empty:
    st.info("No trades yet. Upload a file to get started.")
    st.stop()

# ------------------
# FILTERS
# ------------------

symbols = st.sidebar.multiselect("Filter by Symbol", sorted(df["symbol"].dropna().unique()))
sides = st.sidebar.multiselect("Filter by Side", sorted(df["side"].dropna().unique()))
sessions = st.sidebar.multiselect("Filter by Session", sorted(df["session"].dropna().unique()))

filtered_df = df.copy()
if symbols:
    filtered_df = filtered_df[filtered_df["symbol"].isin(symbols)]
if sides:
    filtered_df = filtered_df[filtered_df["side"].isin(sides)]
if sessions:
    filtered_df = filtered_df[filtered_df["session"].isin(sessions)]

# ------------------
# STATS
# ------------------

total_trades = len(filtered_df)
total_pnl = filtered_df["pnl"].sum()
win_rate = (filtered_df["pnl"] > 0).mean() * 100
avg_rr = filtered_df["rr"].mean()

st.subheader("Performance Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", total_trades)
col2.metric("Total PnL", f"{total_pnl:.2f}")
col3.metric("Win Rate", f"{win_rate:.1f}%")
col4.metric("Avg R:R", f"{avg_rr:.2f}")

# ------------------
# CHARTS
# ------------------

pnl_over_time = px.line(filtered_df, x="entry_dt", y="pnl", title="PnL Over Time")
st.plotly_chart(pnl_over_time, use_container_width=True)

rr_hist = px.histogram(filtered_df, x="rr", nbins=20, title="R:R Distribution")
st.plotly_chart(rr_hist, use_container_width=True)

# ------------------
# TRADE LIST
# ------------------

st.subheader("Trade List")
for idx, tr in filtered_df.iterrows():
    with st.expander(f"{tr.get('entry_dt','')} — {tr.get('symbol','')} ({tr.get('side','')})"):
        st.markdown(f"**Symbol:** {tr.get('symbol','')}, **Entry:** {tr.get('entry_price','')}, **Exit:** {tr.get('exit_price','')}")
        st.markdown(f"**PnL:** {tr.get('pnl','')}, **R:R:** {tr.get('rr','')}, **% Risk:** {tr.get('percent_risk','')}")
        st.markdown(f"**Tags:** {tr.get('tags','')}")
        st.markdown(f"**Context:** {tr.get('context','')}")
        st.markdown(f"**Session:** {tr.get('session','')}")
        st.markdown(f"**Notes:** {tr.get('notes','')}")
        st.markdown(f"**Emotion:** {tr.get('emotion','')}")
        if tr.get('trade_url',''):
            st.markdown(f"[View Chart]({tr.get('trade_url','')})")

# ------------------
# FOOTER
# ------------------

st.caption("Advanced Trading Journal — Powered by Streamlit")

