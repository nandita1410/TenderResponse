import streamlit as st
import pandas as pd
from datetime import datetime
from db import get_connection
import re
import fitz  # PyMuPDF for PDF extraction
import docx   # python-docx for Word files
from VectorDB import get_company_profile_text
from tender_analyzer import (
    retrieve_relevant_documents,
    format_retrieved_context,
    get_bid_decision_and_reasoning_with_rag,
    get_tender_response_outline_with_rag,
    get_estimated_cost_table_with_rag,
    get_win_probability_with_rag,
    get_competitor_list_with_rag,
    get_competitor_win_probabilities_with_rag,
    save_analysis_outputs_to_docx
)
st.set_page_config(page_title="Tender Generation & Tracking", layout="wide")

# ---------- 1. CHECK LOGIN ----------
if "user_email" not in st.session_state:
    st.warning("⚠️ Please log in to access this page.")
    st.stop()

# ---------- 2. GET USER_ID ----------
conn = get_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT user_id, username FROM users WHERE email = %s", (st.session_state["user_email"],))
user = cursor.fetchone()
if not user:
    st.error("User not found. Please log in again.")
    st.stop()

USER_ID = user["user_id"]
USERNAME = user["username"]
st.sidebar.success(f"👤 Logged in as {USERNAME}")

def run_query(query, params=None, fetch=False):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params or ())
    data = cursor.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return data

def extract_text_from_pdf(file):
    pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_doc:
        text += page.get_text("text")
    return text.strip()

def extract_text_from_docx(file):
    document = docx.Document(file)
    return "\n".join([para.text for para in document.paragraphs])

def extract_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

# ---------- Extract Tender Details ----------
def extract_details_from_text(text):
    details = {
        "bid_end_datetime": None,
        "bid_opening_datetime": None,
        "ministry_name": "",
        "department_name": "",
        "organisation_name": "",
    }
    end_date_match = re.search(r"Bid\s*End\s*Date/Time\s*\n?(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})", text)
    if end_date_match:
        details["bid_end_datetime"] = datetime.strptime(end_date_match.group(1), "%d-%m-%Y %H:%M:%S")
    opening_date_match = re.search(r"Bid\s*Opening\s*Date/Time\s*\n?(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})", text)
    if opening_date_match:
        details["bid_opening_datetime"] = datetime.strptime(opening_date_match.group(1), "%d-%m-%Y %H:%M:%S")
    ministry_match = re.search(r"Ministry/State Name\s*\n?([A-Za-z &]+)", text)
    if ministry_match:
        details["ministry_name"] = ministry_match.group(1).strip()
    department_match = re.search(r"Department Name\s*\n?([A-Za-z &]+)", text)
    if department_match:
        details["department_name"] = department_match.group(1).strip()
    org_match = re.search(r"Organisation Name\s*\n?([A-Za-z0-9 .,&-]+)", text)
    if org_match:
        details["organisation_name"] = org_match.group(1).strip()
    return details

# ---------- 3. MAIN PAGE ----------
st.title("Tender Generation & Tracking")
st.subheader("📁 Upload New Tender Documents")
uploaded_file = st.file_uploader("Upload tender document", type=["pdf", "docx", "xlsx"])
extracted_data = {}
text_data = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text_data = extract_text_from_pdf(uploaded_file)
        if not text_data:
            st.warning("⚠️ No text found in this PDF. Please upload a text-based PDF.")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text_data = extract_text_from_docx(uploaded_file)
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]:
        text_data = extract_from_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    extracted_data = extract_details_from_text(text_data)
    st.success("✅ Tender details auto-filled below. Please review before saving.")
    
# ---------- 4. GENERATE INSIGHTS USING AI ----------
# ---------- 4. GENERATE INSIGHTS USING AI ----------
st.subheader("🤖 Generate Tender Insights")

if text_data:
    from tender_analyzer import (
        retrieve_relevant_documents,
        format_retrieved_context,
        get_bid_decision_and_reasoning_with_rag,
        get_tender_response_outline_with_rag,
        get_estimated_cost_table_with_rag
    )

    # Retrieve documents from vector DB and format context
    retrieved_docs = retrieve_relevant_documents(text_data, top_k=5)
    retrieved_context = format_retrieved_context(retrieved_docs)

    colA, colB = st.columns(2)

    # --- 1. BID / NO-BID DECISION ---
    with colA:
        if st.button("🎯 Generate Bid / No-Bid Decision"):
            try:
                with st.spinner("🔄 Generating Bid decision.. please wait..."):
                    decision, explanation = get_bid_decision_and_reasoning_with_rag(
                        tender_text=text_data,
                        company_profile_text=get_company_profile_text(),
                        retrieved_context=retrieved_context
                    )
                st.success(f"🟩 Recommended Decision: **{decision.upper()}**")
                editable_reason = st.text_area("Reasoning (editable):", explanation, height=150, key="reasoning_edit")

                if st.button("💾 Save Decision to Database"):
                    run_query("""
                        INSERT INTO tender_analysis (tender_id, bid_decision, full_tender_response)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE bid_decision=%s, full_tender_response=%s
                    """, (None, decision, editable_reason, decision, editable_reason))
                    st.success("✅ Decision & reasoning saved successfully!")
            except Exception as e:
                st.error(f"Error generating decision: {e}")

    # --- 2. FULL RESPONSE OUTLINE ---
    with colA:
        if st.button("📄 Generate Full Tender Response Outline"):
            try:
                with st.spinner("🔄 Generating Full Tender Response Outline.. please wait..."):
                    outline = get_tender_response_outline_with_rag(
                        tender_text=text_data,
                        retrieved_context=retrieved_context
                    )
                editable_outline = st.text_area("Response Outline (editable):", outline, height=250, key="outline_edit")

                if st.button("💾 Save Response Outline"):
                    run_query("""
                        INSERT INTO tender_analysis (tender_id, full_tender_response)
                        VALUES (%s, %s)
                        ON DUPLICATE KEY UPDATE full_tender_response=%s
                    """, (None, editable_outline, editable_outline))
                    st.success("✅ Tender response saved successfully!")
            except Exception as e:
                st.error(f"Error generating response outline: {e}")

    # --- 3. COST ESTIMATES ---
    with colB:
        if st.button("💰 Generate Cost Estimates (Min/Max)"):
            try:
                with st.spinner("🔄 Generating cost estimates... please wait..."):
                    cost_estimates = get_estimated_cost_table_with_rag(
                        tender_text=text_data,
                        retrieved_context=retrieved_context
                    )

                if cost_estimates:
                    df = pd.DataFrame(cost_estimates)
                    df["min_inr"] = df["min_inr"].astype(float)
                    df["max_inr"] = df["max_inr"].astype(float)
                    df["notes"] = df["notes"].astype(str)

                    total_min = df["min_inr"].sum()
                    total_max = df["max_inr"].sum()

                    st.success(f"💸 Estimated Cost Range: ₹{total_min:,.2f} - ₹{total_max:,.2f}")

                    # Show full editable dataframe
                    edited_df = st.data_editor(
                        df,
                        num_rows="dynamic",
                        key="editable_cost_table",
                        use_container_width=True
                    )

                    # Save edited dataframe back to DB
                    if st.button("💾 Save Cost Estimate"):
                        import json
                        run_query("""
                            INSERT INTO tender_analysis (tender_id, cost_estimate)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE cost_estimate=%s
                        """, (None, json.dumps(edited_df.to_dict(orient="records")), json.dumps(edited_df.to_dict(orient="records"))))
                        st.success("✅ Cost estimate table saved successfully!")

                else:
                    st.warning("⚠️ No cost estimates generated.")
            except Exception as e:
                st.error(f"Error generating cost estimates: {e}")

    # --- 4. WIN PROBABILITY ---
    with colB:
        if st.button("📊 Generate Win Probability"):
            try:
                with st.spinner("Generating win probability..."):
                    my_prob = get_win_probability_with_rag(text_data, retrieved_context)
                    competitors = get_competitor_list_with_rag(text_data, retrieved_context)
                    comp_probs = get_competitor_win_probabilities_with_rag(text_data, competitors, retrieved_context)

                editable_prob = st.number_input("Your Win Probability", 0.0, 1.0, float(my_prob or 0.0), 0.01)
                st.markdown("Competitors List:")
                for c, p in comp_probs.items():
                    st.markdown(f"- **{c}** → {p*100:.1f}%")

                run_query("""
                    INSERT INTO tender_analysis (user_id, win_probability)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE win_probability=%s
                """, (USER_ID, editable_prob, editable_prob))
                st.success("✅ Win probability saved successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
# ---------- 5. FORM TO SAVE ----------
st.subheader("📄 Review & Save Tender Details")
col1, col2 = st.columns(2)
with col1:
    bid_end_datetime = st.text_input(
        "Bid End Date & Time",
        value=extracted_data.get("bid_end_datetime").strftime("%Y-%m-%d %H:%M:%S") if extracted_data.get("bid_end_datetime") else "",
    )
    ministry_name = st.text_input("Ministry Name", value=extracted_data.get("ministry_name", ""))

with col2:
    bid_opening_datetime = st.text_input(
        "Bid Opening Date & Time",
        value=extracted_data.get("bid_opening_datetime").strftime("%Y-%m-%d %H:%M:%S") if extracted_data.get("bid_opening_datetime") else "",
    )
    department_name = st.text_input("Department Name", value=extracted_data.get("department_name", ""))

col3, col4 = st.columns(2)
with col3:
    organisation_name = st.text_input("Organisation Name", value=extracted_data.get("organisation_name", ""))
with col4:
    status = st.selectbox("Tender Status", ["Purchased", "Submitted", "Won", "Lost"], index=0)

bid_amount = st.number_input("Bid Amount (₹)", min_value=0.0, step=100.0)

if st.button("💾 Save Tender to Database"):
    if ministry_name and department_name and organisation_name:
        run_query(
            """
            INSERT INTO tenders
            (user_id, bid_end_datetime, bid_opening_datetime, ministry_name, department_name, organisation_name, status, bid_amount)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                USER_ID,
                extracted_data.get("bid_end_datetime"),
                extracted_data.get("bid_opening_datetime"),
                ministry_name,
                department_name,
                organisation_name,
                status,
                bid_amount
            ),
        )
        st.success("✅ Tender added successfully!")
        st.rerun()
    else:
        st.warning("⚠️ Please ensure all required fields are filled before saving.")

# ---------- 6. UPDATE TENDER ANALYSIS ----------
st.subheader("✏️ Update Tender Analysis")
df = pd.DataFrame(run_query("""
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

if not df.empty:
    selected_tender = st.selectbox("Select Tender", df["ministry_name"].tolist())
    selected_row = df[df["ministry_name"] == selected_tender].iloc[0]
    tender_id = int(selected_row["tender_id"])
    status_options = ["Purchased", "Submitted", "Won", "Lost"]
    current_status = selected_row["status"] if selected_row["status"] in status_options else "Purchased"

    updated_status = st.selectbox(
        "Tender Status",
        status_options,
        index=status_options.index(current_status),
        key=f"status_select_{tender_id}"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        bid_decision = st.selectbox(
            "Bid Decision",
            ["YES", "NO", "REVIEW"],
            index=["YES", "NO", "REVIEW"].index(selected_row["bid_decision"]) if selected_row["bid_decision"] else 2
        )
    with col2:
        cost_estimate = st.number_input("Cost Estimate (₹)", min_value=0.0, value=float(selected_row["cost_estimate"] or 0))
    with col3:
        win_probability = st.slider("Win Probability", 0.0, 1.0, float(selected_row["win_probability"] or 0), 0.01)

    full_response = st.text_area("Full Tender Response", value=selected_row["full_tender_response"] or "")

    if st.button("💾 Save Analysis Updates"):
        run_query("UPDATE tenders SET status = %s WHERE tender_id = %s AND user_id = %s",
                  (updated_status, tender_id, USER_ID))

        exists = run_query("SELECT COUNT(*) AS count FROM tender_analysis WHERE tender_id = %s",
                           (tender_id,), fetch=True)[0]['count']
        if exists:
            run_query("""
                UPDATE tender_analysis
                SET bid_decision=%s, cost_estimate=%s, win_probability=%s, full_tender_response=%s
                WHERE tender_id=%s
            """, (bid_decision, cost_estimate, win_probability, full_response, tender_id))
        else:
            run_query("""
                INSERT INTO tender_analysis (tender_id, bid_decision, cost_estimate, win_probability, full_tender_response)
                VALUES (%s, %s, %s, %s, %s)
            """, (tender_id, bid_decision, cost_estimate, win_probability, full_response))

        st.success("✅ Tender analysis & status updated successfully!")
        st.rerun()

# ---------- 7. FINAL APPROVAL ----------
st.subheader("✅ Final Approval")

# if not df.empty and st.button("📤 Approve and Send to RAG"):
#     approver_email = st.session_state["user_email"]

#     # 1️⃣ Update approval in DB
#     run_query("""
#         UPDATE tender_analysis
#         SET is_approved = TRUE, approved_by = %s, approval_date = %s
#         WHERE tender_id = %s
#     """, (approver_email, datetime.now(), tender_id))
    
#     # 2️⃣ Retrieve tender data from DB for file generation
#     tender_row = run_query("""
#         SELECT t.*, ta.bid_decision, ta.cost_estimate, ta.win_probability, ta.full_tender_response
#         FROM tenders t
#         LEFT JOIN tender_analysis ta ON t.tender_id = ta.tender_id
#         WHERE t.tender_id = %s
#     """, (tender_id,), fetch=True)[0]

#     # 3️⃣ Prepare data for output files
#     analysis_data = {
#         "final_bid_decision": tender_row["bid_decision"] or "REVIEW",
#         "final_decision_reasoning": "",  # Optional: add reasoning if you stored it
#         "tender_response_outline": tender_row["full_tender_response"] or "",
#         "cost_estimates_data": []
#     }

#     # If cost_estimate exists, create DataFrame-like structure
#     if tender_row["cost_estimate"]:
#         try:
#             import json
#             analysis_data["cost_estimates_data"] = json.loads(tender_row["cost_estimate"])
#         except:
#             analysis_data["cost_estimates_data"] = []

#     # 4️⃣ Save outputs to configured output folder
#     try:
#         save_analysis_outputs_to_docx(tender_row["ministry_name"], analysis_data)
#         st.success(f"📬 Tender approved and output files saved to output folder!")
#     except Exception as e:
#         st.error(f"Error generating output files: {e}")

#     st.rerun()
import os
from datetime import datetime

# Define your output path (✅ change this to your actual folder)
OUTPUT_DIR = r"C:\Users\Nandita\OneDrive\Desktop\Tender_Response_System\Sample Response "

if st.button("✅ Approve, Save & Send to RAG"):
    try:
        with st.spinner("📦 Generating final files and updating RAG... please wait..."):

            # Ensure folder exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            tender_id = st.session_state.get("current_tender_id")
            editable_prob = st.session_state.get("editable_prob", 0)

            if not tender_id:
                st.warning("⚠️ Please save or select a tender before approval.")
                st.stop()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1️⃣ Generate Tender Response Outline
            outline_path = os.path.join(OUTPUT_DIR, f"Tender_Response_Outline_{timestamp}.docx")
            save_analysis_outputs_to_docx(tender_id, outline_path)

            # 2️⃣ Save Cost Estimates (Excel)
            if "editable_cost_table" in st.session_state:
                df_cost = st.session_state["editable_cost_table"]
                excel_path = os.path.join(OUTPUT_DIR, f"Estimated_Cost_{timestamp}.xlsx")
                df_cost.to_excel(excel_path, index=False)

            # 3️⃣ Generate Decision Summary
            decision_text = f"Tender {tender_id} approved at {timestamp}. Win probability: {editable_prob * 100:.1f}%"
            decision_path = os.path.join(OUTPUT_DIR, f"Decision_{timestamp}.txt")
            with open(decision_path, "w", encoding="utf-8") as f:
                f.write(decision_text)

            # Update DB
            run_query("""
                UPDATE tender_analysis
                SET status='Approved', approved_at=NOW()
                WHERE tender_id=%s
            """, (tender_id,))

            st.success("✅ Files saved successfully to Output folder!")
            st.toast("All documents exported successfully!", icon="📁")

    except Exception as e:
        st.error(f"Error during approval and export: {e}")


# ---------- 8. DOWNLOAD REPORT ----------
if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Tender Report (CSV)",
        data=csv,
        file_name="tender_report.csv",
        mime="text/csv"
    )

# ---------- LOGOUT ----------
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.switch_page("app.py")
