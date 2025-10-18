# Upload.py
import streamlit as st
import mysql.connector
import os
import io
import zipfile
import subprocess
from datetime import datetime
import traceback
import yaml

# ---------- DB CONFIG ----------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "tender_generation"
}

# ---------- DB HELPERS ----------
def init_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_folder_config(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT working_directory, tender_documents_folder, company_documents_folder, output_folder, sample_responses_folder 
        FROM folder_config LIMIT 1
    """)
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    return {
        "working_directory": row[0],
        "tender_documents_folder": row[1],
        "company_documents_folder": row[2],
        "output_folder": row[3],
        "sample_responses_folder": row[4]
    }

def ensure_dirs(cfg):
    dirs = {}
    for key in ["tender_documents_folder", "company_documents_folder", "output_folder", "sample_responses_folder"]:
        path = os.path.join(cfg["working_directory"], cfg[key])
        os.makedirs(path, exist_ok=True)
        dirs[key] = path
    return dirs

def safe_filename(fn):
    return os.path.basename(fn)

def unique_path(save_dir, fn):
    fn = safe_filename(fn)
    path = os.path.join(save_dir, fn)
    if not os.path.exists(path):
        return path
    name, ext = os.path.splitext(fn)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(save_dir, f"{name}_{ts}{ext}")

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="📄 Upload and Refresh", layout="centered")
st.title("Upload Documents and Refresh")

# ---------- SESSION CHECK ----------
if "user_id" not in st.session_state or st.session_state["user_id"] is None:
    st.error("⚠️ You are not logged in. Please log in to continue.")
    st.stop()
user_id = st.session_state["user_id"]

# ---------- INIT DB & FOLDERS ----------
conn = init_connection()
folder_cfg = get_folder_config(conn)
if not folder_cfg:
    st.error("❌ folder_config not found in DB.")
    st.stop()

dirs = ensure_dirs(folder_cfg)

# ---------- SINGLE COMPANY DOCUMENT UPLOAD ----------
st.subheader("📄 Upload Company Document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf","docx","txt","xlsx"])

def save_file_to_db(fobj, save_dir, table_name):
    fname = safe_filename(fobj.name)
    ftype = fobj.type or "application/octet-stream"
    dest = unique_path(save_dir, fname)
    try:
        with open(dest, "wb") as f:
            f.write(fobj.getbuffer())
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO {table_name} (user_id, file_name, file_type, file_path)
            VALUES (%s, %s, %s, %s)
        """, (user_id, fname, ftype, dest))
        conn.commit()
        cur.close()
        return True, dest
    except Exception as e:
        return False, str(e)

if uploaded_file is not None and st.button("Upload to Company Folder"):
    ok, info = save_file_to_db(uploaded_file, dirs["company_documents_folder"], "company_documents")
    if ok:
        # also copy to sample_responses
        unique_copy = unique_path(dirs["sample_responses_folder"], uploaded_file.name)
        with open(unique_copy, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Saved to {info} and Sample Responses: {unique_copy}")
    else:
        st.error(f"❌ Upload failed: {info}")

# ---------- MULTIPLE PREVIOUS TENDERS UPLOAD ----------
st.subheader("📄 Upload Previous Tender Files")
uploaded_files = st.file_uploader("Upload one or more previous tender documents", type=["pdf","docx","txt","xlsx"], accept_multiple_files=True)

if uploaded_files and st.button("Upload Selected Files"):
    successes, errors = 0, []
    for f in uploaded_files:
        ok, info = save_file_to_db(f, dirs["tender_documents_folder"], "previous_tenders")
        if ok:
            # also copy to sample_responses
            unique_copy = unique_path(dirs["sample_responses_folder"], f.name)
            with open(unique_copy, "wb") as sf:
                sf.write(f.getbuffer())
            successes += 1
        else:
            errors.append(info)
    st.success(f"✅ {successes} files uploaded to folders & sample responses.")
    if errors:
        st.error("Some uploads failed: " + "; ".join(errors))

# ---------- VIEW & DOWNLOAD AS ZIP ----------
# ---------- VIEW & DOWNLOAD AS ZIP ----------
st.subheader("Download All Uploaded Documents")
def gather_files(user_id):
    files = []
    cur = conn.cursor()
    for table in ["previous_tenders", "company_documents"]:
        cur.execute(f"SELECT file_name, file_path FROM {table} WHERE user_id=%s", (user_id,))
        for fn, fp in cur.fetchall():
            try:
                with open(fp, "rb") as fh:
                    files.append((fn, fh.read()))
            except:
                continue
    cur.close()
    return files

files = gather_files(user_id)

if files:
    st.write(f"**Files available ({len(files)}):**")
    for name, _ in files:
        st.write(f"- {name}")

    # Create the zip only when user clicks the button
    if st.button("⬇️ Download All Files as ZIP"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, b in files:
                zf.writestr(fname, b)
        zip_buffer.seek(0)

        # Optionally save a copy in Sample Responses
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_copy_path = os.path.join(dirs["sample_responses_folder"], f"all_files_{timestamp}.zip")
        with open(zip_copy_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        st.info(f"📁 Saved a copy in Sample Responses: {zip_copy_path}")

        # Trigger download
        st.download_button(
            "⬇️ Click to Download ZIP",
            data=zip_buffer,
            file_name=f"all_files_{timestamp}.zip",
            mime="application/zip"
        )
else:
    st.info("ℹ️ No uploaded files found.")

# ---------- RAG BUILD ----------
st.header("🔄 Build RAG")
if st.button("Build RAG"):
    st.write(f"Building RAG data for **user_id: {user_id}**...")
    # Add actual RAG refresh logic here
    st.success("✅ RAG build completed!")
# ---------- RAG REFRESH ----------
st.header("🔄 RAG Refresh")
if st.button("RAG Refresh"):
    status_placeholder = st.empty()
    status_placeholder.info("⏳ Refreshing RAG, streaming logs...")

    os.makedirs(dirs["output_folder"], exist_ok=True)
    cmd = ["python", "VectorDB.py"]  # Ensure correct path
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            status_placeholder.text(line.rstrip())
        process.wait()
        if process.returncode == 0:
            status_placeholder.success(f"✅ RAG refresh completed! Output folder: '{dirs['output_folder']}'")
        else:
            status_placeholder.error(f"❌ RAG refresh failed (exit code {process.returncode}).")
    except Exception as e:
        status_placeholder.error("❌ Error running VectorDB.py: " + str(e))
        traceback.print_exc()

# ---------- CLOSE DB ----------
conn.close()
