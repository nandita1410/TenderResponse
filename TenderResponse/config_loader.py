# config_loader.py
import yaml
import mysql.connector
import sys

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "tender_generation"
}

MASTER_YML = "master.yml"

def init_connection():
    return mysql.connector.connect(**DB_CONFIG)

def load_master_yml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def upsert_folder_config(conn, cfg):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM folder_config LIMIT 1")
    r = cursor.fetchone()

    data = (
        cfg.get("working_directory"),
        cfg.get("tender_documents_folder"),
        cfg.get("company_documents_folder"),
        cfg.get("output_folder"),
        cfg.get("sample_responses_folder"),
        cfg.get("company_profile_filename"),
        cfg.get("company_turnover_filename"),
    )

    if r:
        cursor.execute("""
            UPDATE folder_config SET 
                working_directory=%s, tender_documents_folder=%s, company_documents_folder=%s,
                output_folder=%s, sample_responses_folder=%s, company_profile_filename=%s,
                company_turnover_filename=%s, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
        """, (*data, r[0]))
    else:
        cursor.execute("""
            INSERT INTO folder_config
                       
            (working_directory, tender_documents_folder, company_documents_folder, output_folder,
             sample_responses_folder, company_profile_filename, company_turnover_filename)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, data)

    conn.commit()
    cursor.close()

def main():
    try:
        cfg = load_master_yml(MASTER_YML)
    except Exception as e:
        print("Error reading master.yml:", e)
        sys.exit(1)

    conn = init_connection()
    try:
        upsert_folder_config(conn, cfg)
        print("✅ folder_config table updated successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
