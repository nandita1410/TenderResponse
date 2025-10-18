import mysql.connector
import streamlit as st
import os

def get_connection():
    """
    Establishes a connection to the MySQL database.
    Works both on Streamlit Cloud (st.secrets) and locally (os.environ).
    """
    try:
        # --- Try using Streamlit secrets first ---
        if "mysql" in st.secrets:
            db_config = st.secrets["mysql"]
            conn = mysql.connector.connect(
                host=db_config.get("server", "localhost"),
                port=db_config.get("port", 3306),
                user=db_config.get("user"),
                password=db_config.get("password"),
                database=db_config.get("database")
            )
        else:
            # --- Fallback for local development ---
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "3306")),
                user=os.getenv("DB_USER", "root"),
                password=os.getenv("DB_PASSWORD", ""),
                database=os.getenv("DB_NAME", "tender_generation")
            )
        return conn

    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()


def upload_file_to_db(file_path, doc_type, folder_name=None):
    """Uploads a file into the 'documents' table as a BLOB."""
    conn = get_connection()
    cursor = conn.cursor()

    with open(file_path, "rb") as f:
        file_data = f.read()

    cursor.execute("""
        INSERT INTO documents (doc_name, doc_type, folder_name, content)
        VALUES (%s, %s, %s, %s)
    """, (os.path.basename(file_path), doc_type, folder_name, file_data))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Uploaded {file_path} as {doc_type}")


def fetch_file_from_db(doc_name, save_path):
    """Fetches a file from the DB and saves it locally."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT content FROM documents WHERE doc_name=%s", (doc_name,))
    result = cursor.fetchone()

    if result:
        with open(save_path, "wb") as f:
            f.write(result[0])
        print(f"✅ Saved {doc_name} to {save_path}")
    else:
        print(f"⚠️ Document {doc_name} not found in DB.")

    cursor.close()
    conn.close()
