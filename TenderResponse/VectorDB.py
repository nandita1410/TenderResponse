import os
import json
import yaml
import pypdf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import List, Dict, Any, Optional
import mysql.connector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---------------- Configuration ----------------
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_SIZE = 768  # Dimension for Gemini embeddings

CONFIG: Dict[str, Any] = {}
VECTOR_DB: List[Dict[str, Any]] = []

# ---------------- Load Config ----------------
def load_config(config_path: str = 'master.yml'):
    global CONFIG
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
        working_dir = CONFIG.get('working_directory', os.getcwd())
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)
        print(f"[INFO] Working directory set to: {os.getcwd()}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Loading config failed: {e}")

# ---------------- Embeddings ----------------
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    if not API_KEY:
        raise ValueError("API_KEY not found. Please provide it.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=API_KEY)

def embed_text(text: str) -> List[float]:
    try:
        model = get_embedding_model()
        return model.embed_query(text)
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return []

# ---------------- Document Extraction ----------------
def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    if not os.path.exists(pdf_path):
        return None
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"[ERROR] Reading PDF {pdf_path}: {e}")
        return None
    return text

def extract_costs_from_excel(excel_path: str) -> List[Dict[str, Any]]:
    costs = []
    if not os.path.exists(excel_path):
        return costs
    try:
        df = pd.read_excel(excel_path)
        for _, row in df.iterrows():
            if "total" not in str(row.get('Cost Component', '')).lower():
                costs.append({
                    "component": row.get('Cost Component'),
                    "min_inr": row.get('Minimum Estimate (INR)'),
                    "max_inr": row.get('Maximum Estimate (INR)'),
                    "notes": row.get('Notes')
                })
    except Exception as e:
        print(f"[ERROR] Reading Excel {excel_path}: {e}")
    return costs

# ---------------- VectorDB ----------------
def add_to_vector_db(text: str, dtype: str, source: str):
    embedding = embed_text(text)
    if embedding:
        VECTOR_DB.append({
            "text": text,
            "embedding": embedding,
            "type": dtype,
            "source": source
        })

# ---------------- Ensure Sample Responses folder in DB ----------------
def ensure_sample_response_folder():
    folder_name = CONFIG.get('sample_responses_folder', 'Sample Responses')
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"[INFO] Sample Responses folder ensured at: {folder_path}")

    # Log in MySQL
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="root", database="tender_generation"
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sample_response (
                id INT AUTO_INCREMENT PRIMARY KEY,
                folder_name VARCHAR(255),
                folder_path VARCHAR(500),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("SELECT id FROM sample_response WHERE folder_name=%s", (folder_name,))
        if not cur.fetchone():
            cur.execute("INSERT INTO sample_response(folder_name, folder_path, created_at) VALUES (%s,%s,%s)",
                        (folder_name, folder_path, datetime.now()))
            conn.commit()
            print("[INFO] Sample Responses folder logged in MySQL.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Logging Sample Responses folder failed: {e}")

# ---------------- Save Output Files to DB ----------------
def save_output_files_to_db(user_id: int, output_path: str):
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="root", database="tender_generation"
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS generated_outputs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                file_type VARCHAR(50),
                file_data LONGBLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        for file_name in os.listdir(output_path):
            file_path = os.path.join(output_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    file_data = f.read()
                ftype = os.path.splitext(file_name)[1].replace(".", "")
                cur.execute("""
                    INSERT INTO generated_outputs (user_id, file_name, file_type, file_data)
                    VALUES (%s,%s,%s,%s)
                """, (user_id, file_name, ftype, file_data))
                conn.commit()
                print(f"[INFO] Saved {file_name} to DB.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Saving Output folder files to DB failed: {e}")

# ---------------- Create and Save VectorDB ----------------
def create_and_save_vector_db(user_id: int = 1):
    global VECTOR_DB
    load_config()
    ensure_sample_response_folder()

    # Paths
    sample_responses_path = os.path.join(os.getcwd(), CONFIG.get('sample_responses_folder', 'Sample Responses'))
    company_docs_path = os.path.join(os.getcwd(), CONFIG.get('company_documents_folder', 'Company Docs'))
    output_path = os.path.join(os.getcwd(), CONFIG.get('output_folder', 'Outputs'))
    os.makedirs(output_path, exist_ok=True)

    # --- Sample Responses ---
    if os.path.isdir(sample_responses_path):
        for folder_name in os.listdir(sample_responses_path):
            folder_path = os.path.join(sample_responses_path, folder_name)
            if os.path.isdir(folder_path):
                tender_text = extract_text_from_pdf(os.path.join(folder_path, "tender_document.pdf"))
                outline_text = extract_text_from_pdf(os.path.join(folder_path, "Tender_Response_Outline.pdf"))
                decision_text = extract_text_from_pdf(os.path.join(folder_path, "Decision.pdf"))
                cost_data = extract_costs_from_excel(os.path.join(folder_path, "Estimated_Cost.xlsx"))

                if tender_text: add_to_vector_db(tender_text, "sample_tender", folder_name)
                if outline_text: add_to_vector_db(outline_text, "sample_outline", folder_name)
                if decision_text: add_to_vector_db(decision_text, "sample_decision", folder_name)
                if cost_data: add_to_vector_db(json.dumps(cost_data), "sample_costs", folder_name)

    # --- Company Docs ---
    company_profile_text = extract_text_from_pdf(os.path.join(company_docs_path, CONFIG.get('company_profile_filename', '')))
    if company_profile_text:
        add_to_vector_db(company_profile_text, "company_profile", CONFIG.get('company_profile_filename', ''))

    company_turnover_text = extract_text_from_pdf(os.path.join(company_docs_path, CONFIG.get('company_turnover_filename', '')))
    if company_turnover_text:
        add_to_vector_db(company_turnover_text, "company_turnover", CONFIG.get('company_turnover_filename', ''))

    # --- Save VectorDB JSON ---
    vector_db_file = os.path.join(output_path, 'vector_db.json')
    with open(vector_db_file, "w", encoding="utf-8") as f:
        json.dump(VECTOR_DB, f, indent=2)
    print(f"[INFO] VectorDB saved to {vector_db_file}")

    # --- Save vector embeddings (.npy) and metadata (.joblib) ---
    embeddings = np.array([item["embedding"] for item in VECTOR_DB if item.get("embedding")])
    metadata = [ {k: v for k, v in item.items() if k != "embedding"} for item in VECTOR_DB ]
    np.save(os.path.join(output_path, "vector_embeddings.npy"), embeddings)
    joblib.dump(metadata, os.path.join(output_path, "vector_db_metadata.joblib"))
    print(f"[INFO] Embeddings (.npy) and metadata (.joblib) saved in {output_path}")

    # --- Save all Output folder files to DB ---
    save_output_files_to_db(user_id, output_path)

# ---------------- Get Company Profile Text ----------------
def get_company_profile_text():
    """
    Retrieves company profile text from the company documents folder.
    Uses extract_text_from_pdf if the profile file exists.
    """
    
    try:
        # Load config
        CONFIG_PATH = os.path.join(os.getcwd(), "master.yml")
        import yaml
        with open(CONFIG_PATH, "r") as f:
            CONFIG = yaml.safe_load(f)

        company_docs_path = os.path.join(os.getcwd(), CONFIG.get('company_documents_folder', 'Company Docs'))
        profile_filename = CONFIG.get('company_profile_filename', '')

        profile_path = os.path.join(company_docs_path, profile_filename)

        if not profile_filename or not os.path.exists(profile_path):
            print("[WARN] Company profile file not found.")
            return ""
        text = extract_text_from_pdf(profile_path)
        return text or ""
    except Exception as e:
        print(f"[ERROR] Could not retrieve company profile text: {e}")
        return ""

# ---------------- Main ----------------
if __name__ == "__main__":
    create_and_save_vector_db()
