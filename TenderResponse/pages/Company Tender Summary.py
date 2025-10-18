import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from db import get_connection
import re
 
st.set_page_config(page_title="📊 Tender Dashboard", layout="wide")
 
# ---------- 1. CHECK LOGIN ----------
if "user_email" not in st.session_state:
    st.warning("⚠️ Please log in to view your dashboard.")
    st.stop()
def run_query(query, params=None, fetch=False):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params or ())
    data = cursor.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return data 
# ---------- 2. GET LOGGED-IN USER DETAILS ----------
conn = get_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT user_id, username FROM users WHERE email = %s", (st.session_state["user_email"],))
user = cursor.fetchone()
 
if not user:
    st.error("User not found. Please log in again.")
    st.stop()

USER_ID = user["user_id"]
USERNAME = user["username"]
user_email = st.session_state.get("user_email")
user_id = st.session_state.get("user_id")
 

st.sidebar.success(f"👤 Logged in as {USERNAME}")
 
cursor.execute("""
        SELECT t.*, u.username
        FROM tenders t
        JOIN users u ON t.user_id = u.user_id
        WHERE t.user_id = %s """, (USER_ID,))
 
data = cursor.fetchall()
conn.close()
 
# Convert to DataFrame
df = pd.DataFrame(data)
 
if df.empty:
    st.warning("⚠️ No tenders found for this user.")
    st.stop()
 
# Ensure proper datetime conversion
df["bid_end_datetime"] = pd.to_datetime(df["bid_end_datetime"])
df["bid_opening_datetime"] = pd.to_datetime(df["bid_opening_datetime"])

if "@" in user_email:
    raw_name = user_email.split("@")[0]
    user_name = re.sub(r'[^A-Za-z]', '', raw_name)
else:
    user_name = user_email
# ---------- 4. CHARTS ---------
st.title(f"📊 Dashboard for {user_name}")

# ---------- 5️. View Tenders (User-Specific) ----------
st.subheader("Tender Tracking & Analysis")
 
df1 = pd.DataFrame(run_query("""
    SELECT
        t.tender_id,
        t.ministry_name,
        t.organisation_name,
        t.department_name,
        t.status,
        ta.bid_decision,
        t.bid_amount,
        ta.cost_estimate,
        ta.win_probability,
        ta.full_tender_response,
        ta.is_approved,
        ta.approved_by,
        ta.approval_date
        
    FROM tenders t
    LEFT JOIN tender_analysis ta ON t.tender_id = ta.tender_id
    WHERE t.user_id = %s
    ORDER BY t.tender_id DESC
""", (USER_ID,), fetch=True))
 
if df1.empty:
    st.warning("📭 No tenders found for your account.")
else:
    st.dataframe(df1, use_container_width=True)
 
# ---------------- Chart 1: Pie Chart for Tender Status ----------------
status_counts = df["status"].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]
 
fig_pie = px.pie(
    status_counts,
    names="Status",
    values="Count",
    title="Tender Status Distribution",
    color="Status",
)
st.plotly_chart(fig_pie, use_container_width=True)
 
# ---------------- Chart 2: Bar Chart - Bid Amount by Organisation ----------------
st.subheader("💰 Bid Amount by Organisation")
fig_bar = px.bar(
    df,
    x="organisation_name",
    y="bid_amount",
    color="status",
    title="Bid Amount by Organisation",
    labels={"organisation_name": "Organisation", "bid_amount": "Bid Amount (₹)"}
)
st.plotly_chart(fig_bar, use_container_width=True)
 
# ---------------- Chart 3: Line Chart - Timeline of Tenders ----------------
st.subheader("📆 Tender Activity Over Time")
timeline_df = df.groupby(df["bid_end_datetime"].dt.date).size().reset_index(name="Tender Count")
 
fig_line = px.line(
    timeline_df,
    x="bid_end_datetime",
    y="Tender Count",
    title="Tenders Over Time",
    markers=True,
)
st.plotly_chart(fig_line, use_container_width=True)
 
# ---------------- Chart 8: Bar Chart by Ministry ----------------
st.subheader("🏛 Bid Amount by Ministry")
fig_ministry = px.bar(
    df,
    x="ministry_name",
    y="bid_amount",
    color="status",
    title="Bid Amount by Ministry",
    labels={"ministry_name": "Ministry", "bid_amount": "Bid Amount (₹)"}
)
st.plotly_chart(fig_ministry, use_container_width=True)
# ---------------- Chart 4: Average Bid Amount by Status ----------------
st.subheader("📈 Average Bid Amount by Status")

avg_bid_df = df.groupby("status")["bid_amount"].mean().reset_index()

fig_avg_bid = px.bar(
    avg_bid_df,
    x="status",
    y="bid_amount",
    color="status",
    title="Average Bid Amount by Tender Status",
    labels={"status": "Status", "bid_amount": "Average Bid Amount (₹)"},
    text_auto=".2s"
)
fig_avg_bid.update_traces(textposition="outside")
st.plotly_chart(fig_avg_bid, use_container_width=True)
# ---------------- Chart 6: Sunburst Chart - Tender Hierarchy ----------------
st.subheader("🌐 Tender Hierarchy: Ministry → Department → Organisation")

fig_sunburst = px.sunburst(
    df,
    path=["ministry_name", "department_name", "organisation_name"],
    values="bid_amount",
    color="status",
    title="Tender Distribution by Ministry, Department & Organisation"
)
st.plotly_chart(fig_sunburst, use_container_width=True)


if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.switch_page("app.py")