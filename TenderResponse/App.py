import streamlit as st
from db import get_connection
import os

# Page config
st.set_page_config(page_title="Tender Response System", layout="wide")

# --- CSS for centering and styling ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        .forgot-link{
            text-align:center;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .center-container .stButton>button {
            margin-top: 10px;
            width: 200px; /* Optional: make buttons same width */
        }

        .center-container a {
            margin-top: 10px;
            text-decoration: none;
            color: #0f4c81;
            font-weight: bold;
        }

        .center-container a:hover {
            text-decoration: underline;
        }

        .login-form {
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-items: center;
            width: 100%;
            max-width: 300px; /* Makes form narrower */
        }

        .login-form input {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "redirect" not in st.session_state:
    st.session_state["redirect"] = False

# --- Columns to center content ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='center-container'>", unsafe_allow_html=True)

    # Logo
    IMAGE_FILE = r"C:\Users\Nandita\OneDrive\Desktop\TenderResponse\Picture1.jpg"
    if os.path.exists(IMAGE_FILE):
        st.image(IMAGE_FILE, width=150)

    # Headings
    st.markdown("<h1 style='text-align:center;'>AI-Driven Tender Response System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Log Into Your Account</h3>", unsafe_allow_html=True)

    # --- Login Form ---
    with st.form(key="login_form"):
        st.markdown("<div class='login-form'>", unsafe_allow_html=True)
        email = st.text_input("Email or phone number")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Log In")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Forgot Password Link ---
    st.markdown('<a href="#" class="forgot-link">Forgot Password?</a>', unsafe_allow_html=True)

    # --- Create Account Button ---
    if st.button("Create New Account"):
        st.info("Redirect to create account page (handle this).")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Handle Login ---
    if login_button:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM users WHERE email=%s AND password=%s"
        cursor.execute(query, (email, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = email
            st.session_state["user_id"] = user["user_id"]
            st.success("Login Successful ✅")
            st.session_state["redirect"] = True  # Flag to redirect
        else:
            st.error("Invalid credentials ❌")

# --- Redirect after login ---
if st.session_state.get("redirect"):
    st.session_state["redirect"] = False
    st.switch_page("pages/Company Tender Summary.py")

# --- Redirect if already logged in ---
if st.session_state.get("logged_in"):
    st.switch_page("pages/Company Tender Summary.py")
