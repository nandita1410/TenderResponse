# db_utils.py
import mysql.connector
import streamlit as st
import os

def get_connection():
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["server"],
            port=st.secrets["mysql"]["port"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()
        
def upload_file_to_db(file_path, doc_type, folder_name=None):
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
