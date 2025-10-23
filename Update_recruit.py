# # -*- coding: utf-8 -*-
# Personato TalentLens — COMPLETE (+ Vacancy Upload/Paste → Auto-Classify → Auto-Match)
# Dark UI, EN/NL, Sector→Vacancy, Batch uploads, OCR toggle + tuning, Single chat with threads (archive/delete),
# Grounded Q&A + Enhanced Narrative, Custom Factors + What-if, SHAP (global/sector & per-candidate),
# Explorer, Outreach, Feature Snapshot chart on Dashboard, and Batch CV Narratives.
#
# New/Updated:
# - Upload/paste job vacancy (PDF/DOCX/TXT or raw text) with OCR fallback
# - Automatic sector classification + closest job title suggestion
# - Dynamic mapping to Dashboard sector & vacancy selectors (safe session-state)
# - Vacancy text automatically included in candidate analysis + conversational Q&A
# - “Matched vacancy” badge on Dashboard (auto/ manual + similarity score)
# - Batch: instant narratives per uploaded CV (keeps all features intact)

import os, re, io, json, uuid
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
import shap


# Retrieval (lightweight)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional deps
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    import docx
except Exception:
    docx = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# OCR deps (optional)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image, ImageOps  # noqa
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Extra PDF fallback
try:
    import pypdf
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False
# --- Streamlit page & theme ---
# --- Streamlit page & theme ---
st.set_page_config(page_title="CynthAI© - Personato TalentLens", page_icon="🤝", layout="wide")

# ===== 🎨 THEME COLOR VARIABLES (for charts & metrics) =====
PRIMARY_BG = "#000000"       # Crowe black
TEXT_LIGHT = "#EAEAEA"       # light gray text
ACCENT_RED = "#E50046"       # TalentLens red
GOLD = "#FFD700"             # Crowe gold (for optional highlights)

# ===== 🔐 CynthAI© TalentLens — Full Auth System (Crowe Black + Gold, Centered Layout) =====
import streamlit as st
import hashlib, json, os
from datetime import datetime
from pathlib import Path
from PIL import Image

# -------------------- USER STORAGE --------------------
USER_FILE = Path("users.json")

# ✅ Persistent user session + disk reload
if "users" not in st.session_state:
    if USER_FILE.exists():
        with open(USER_FILE, "r") as f:
            st.session_state.users = json.load(f)
    else:
        st.session_state.users = {}

USERS = st.session_state.users


# -------------------- UTILS --------------------
def hash_password(pw: str) -> str:
    """Securely hash passwords."""
    return hashlib.sha256(pw.encode()).hexdigest()

def save_users():
    """Save updated user list to disk."""
    with open(USER_FILE, "w") as f:
        json.dump(st.session_state.users, f, indent=2)

def signup_user(email, name, password):
    """Handle new user sign-up logic with auto-login."""
    if not email or not name or not password:
        st.warning("⚠️ Please fill in all fields.")
    elif email in st.session_state.users:
        st.error("This email is already registered.")
    elif len(password) < 5:
        st.warning("Password must be at least 5 characters long.")
    else:
        st.session_state.users[email] = {
            "name": name.strip(),
            "password": hash_password(password),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_users()
        st.success(f"✅ Welcome, {name.split()[0]}! Account created successfully.")

        # ✅ Auto-login after signup
        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.session_state.user_name = name
        st.session_state.login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()

def login_user(email, password):
    """Validate login credentials."""
    if email not in st.session_state.users:
        st.error("❌ No account found. Please sign up.")
    elif st.session_state.users[email]["password"] != hash_password(password):
        st.error("❌ Invalid password.")
    else:
        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.session_state.user_name = st.session_state.users[email]["name"]
        st.session_state.login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()

def logout_user():
    """Log out the current user."""
    for k in ["logged_in", "user_email", "user_name", "login_time"]:
        st.session_state.pop(k, None)
    st.rerun()

def reset_password(email, new_pw):
    """Reset password if user exists."""
    if email not in st.session_state.users:
        st.error("❌ No account found for this email.")
    elif len(new_pw) < 5:
        st.warning("Password too short.")
    else:
        st.session_state.users[email]["password"] = hash_password(new_pw)
        save_users()
        st.success("✅ Password reset successful. You can now log in.")
        st.rerun()


# -------------------- PAGE STYLE --------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    background-color: #000 !important;
    color: #EAEAEA !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.login-container {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.login-card {
    background-color: #0E0E0E;
    border: 1px solid rgba(255,215,0,0.25);
    box-shadow: 0 0 35px rgba(255,215,0,0.15);
    border-radius: 16px;
    padding: 2rem 1.4rem 2.4rem 1.4rem;
    width: 360px;
    text-align: center;
    color: #EAEAEA;
}

/* Center tabs, compact size */
[data-baseweb="tab-list"] {
    justify-content: center !important;
    border-bottom: 1px solid rgba(255,215,0,0.25) !important;
    width: fit-content !important;
    margin: 0.5rem auto 1rem auto !important;
}
[data-baseweb="tab"] {
    color: gold !important;
    font-weight: 500;
    min-width: 90px !important;
}
[data-baseweb="tab"]:hover {
    color: #FFD700 !important;
}

/* Input field design */
.stTextInput > div > div > input {
    background-color: #1A1A1A !important;
    color: #EAEAEA !important;
    border: 1px solid #555 !important;
    border-radius: 6px !important;
    width: 240px !important;
    margin: 6px auto !important;
    display: block !important;
    text-align: left !important;
}
div[data-testid="stVerticalBlock"] div:has(input) {
    background: none !important;
}

/* Compact, golden buttons */
.stButton>button {
    background: linear-gradient(90deg, #FFD700, #E6BE00);
    color: black;
    font-weight: 600;
    border-radius: 6px;
    width: 240px !important;
    margin: 10px auto 0 auto !important;
    transition: all 0.3s ease-in-out;
    border: none !important;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #FFE44D, #FFD700);
    transform: scale(1.03);
}

/* Text & captions */
h1, h2, h3, h4, label { color: gold !important; }
a { color: white !important; text-decoration: none !important; }
a:hover { text-decoration: underline !important; color: #FFD700 !important; }
small, .stMarkdown p, .stCaption, .stMarkdown a, .stMarkdown { color: white !important; font-size: 0.8rem !important; }

/* Input focus effect */
.stTextInput > div > div > input:focus {
    outline: none !important;
    border: 1px solid gold !important;
    box-shadow: 0 0 8px gold !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# -------------------- AUTH UI --------------------
if not st.session_state.logged_in:
    st.markdown("<div class='login-container'><div class='login-card'>", unsafe_allow_html=True)

    # --- Hosted Logo (always available on Streamlit Cloud) ---
    logo_url = "https://raw.githubusercontent.com/dade85/hr_recruitment_update/6e159ada1056008ae6fb7ddf27e94361e19f5881/CynthAI_Logo.png"

    try:
        st.image(logo_url, width=200)
    except Exception as e:
        # Secondary fallback (optional placeholder)
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Recruitee_logo.png", width=180)
        st.caption("CynthAI© TalentLens")

    # --- Portal Title ---
    st.markdown("## CynthAI© TalentLens Portal")

    # --- Authentication Tabs ---
    

  
    tabs = st.tabs(["🔑 Login", "📝 Sign Up", "🔁 Forgot Password"])

    # --- LOGIN TAB ---
    with tabs[0]:
        st.markdown("### Sign in to your account")
        email = st.text_input("📧 Email", key="login_email")
        pw = st.text_input("🔒 Password", type="password", key="login_pw")
        if st.button("Sign In", use_container_width=True, key="login_btn"):
            login_user(email, pw)
        st.caption("No account yet? Switch to **Sign Up**.")

    # --- SIGN UP TAB ---
    with tabs[1]:
        st.markdown("### Create an account")
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            name = st.text_input("👤 Full Name", key="signup_name")
        with col2:
            email_new = st.text_input("📧 Email", key="signup_email")
        pw_new = st.text_input("🔒 Password", type="password", key="signup_pw")
        if st.button("Create Account", use_container_width=True, key="signup_btn"):
            signup_user(email_new, name, pw_new)
        st.caption("Already registered? Go to **Login**.")

    # --- FORGOT PASSWORD TAB ---
    with tabs[2]:
        st.markdown("### Reset your password")
        reset_email = st.text_input("📧 Registered Email", key="reset_email")
        new_pw = st.text_input("🔒 New Password", type="password", key="reset_pw")
        if st.button("Reset Password", use_container_width=True, key="reset_btn"):
            reset_password(reset_email, new_pw)
        st.caption("Back to **Login** after reset.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# -------------------- DASHBOARD HEADER --------------------
if st.session_state.get("logged_in", False):
    with st.sidebar:
        st.success(f"👤 {st.session_state.user_name}")
        st.caption(f"🕒 Logged in: {st.session_state.login_time}")
        if st.button("🚪 Logout", key="logout_btn"):
            logout_user()

    st.title(f"Welcome, {st.session_state.user_name} 👋🏿")
    st.markdown("""
    This is your personalized **CynthAI© TalentLens Dashboard**.
    
    📘 **Need help?**  
    👉 [Go to Functional Documentation](#functional-documentation-tab)
    """)

    st.divider()


################################################### Main ##############################################################
# ===== ✅ UNIFIED STYLE BLOCK SUITE (CYNTHAI© TALENTLENS THEME) =====
st.markdown("""
<style>

/* ==============================================================
   🎨 GLOBAL BASE THEME
   ============================================================= */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
  background: #000 !important;
  color: #EAEAEA !important;
  font-family: "Poppins", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", Arial, sans-serif;
  height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow-x: hidden !important;
}

/* Accent colors */
:root {
  --gold: #FFD700;
  --accent-red: #E50046;
  --accent-green: #00C853;
  --light-text: #EAEAEA;
  --dark-bg: #0E0E0E;
}

/* Headings and general text */
h1, h2, h3, h4, h5, h6 { color: var(--gold) !important; }
p, span, small, label { color: var(--light-text) !important; opacity: 0.95 !important; }

/* Links */
a, .stMarkdown a { color: white !important; text-decoration: none !important; }
a:hover, .stMarkdown a:hover { color: var(--gold) !important; text-decoration: underline !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: #1A2634 !important;
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Metrics and tables */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
.stDataFrame, .stTable, .stMarkdown table, .stMarkdown th, .stMarkdown td {
  color: white !important;
}

/* Chat message text */
[data-testid="stChatMessage"] * { color: white !important; opacity: 0.98 !important; }
[data-testid="stChatMessage"] a:hover { color: var(--accent-red) !important; }

/* Tabs */
button[data-baseweb="tab"] { color: white !important; }
button[data-baseweb="tab"]:hover,
button[aria-selected="true"][data-baseweb="tab"] {
  color: var(--gold) !important;
  border-bottom: 2px solid var(--gold) !important;
}

/* ============================================================== 
   📁 UPLOADERS (Main + Sidebar)
   ============================================================== */
[data-testid="stFileUploaderDropzone"] {
  color: #000 !important;
  background: #fff !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
  color: #000 !important;
  opacity: 1 !important;
}
[data-testid="stFileUploaderDropzone"] svg {
  color: #000 !important;
  fill: #000 !important;
  stroke: #000 !important;
}
/* Sidebar variant */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: #fff !important;
  color: #000 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label span,
[data-testid="stSidebar"] [data-testid="stCheckbox"] p {
  color: white !important;
  opacity: 1 !important;
}

/* ============================================================== 
   🟢 GREEN CREATE BUTTON (Scoped via #new-thread-form-wrap)
   ============================================================== */
#new-thread-form-wrap [data-testid="baseButton-secondaryFormSubmit"],
#new-thread-form-wrap [data-testid="baseButton-primaryFormSubmit"],
#new-thread-form-wrap [data-testid="baseButton-button"],
#new-thread-form-wrap button[kind="secondaryFormSubmit"],
#new-thread-form-wrap form#new_thread_form button[type="submit"],
#new-thread-form-wrap form[data-testid="stForm"] button[type="submit"] {
  background: var(--accent-green) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  padding: 0.5rem 0.9rem !important;
  box-shadow: none !important;
}
#new-thread-form-wrap button:hover {
  background: #00E676 !important;
  filter: brightness(1.05) !important;
}

/* ============================================================== 
   🟡 LOGIN / SIGNUP / AUTH CONTAINERS
   ============================================================== */
.login-container {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}
.login-card {
  background-color: var(--dark-bg);
  border: 1px solid rgba(255,215,0,0.25);
  box-shadow: 0 0 35px rgba(255,215,0,0.15);
  border-radius: 16px;
  padding: 2rem 1.4rem 2.4rem;
  width: 360px;
  text-align: center;
  color: var(--light-text);
}

/* Centered, shorter tabs */
[data-baseweb="tab-list"] {
  justify-content: center !important;
  border-bottom: 1px solid rgba(255,215,0,0.25) !important;
  width: fit-content !important;
  margin: 0.5rem auto 1rem auto !important;
}
[data-baseweb="tab"] {
  color: var(--gold) !important;
  font-weight: 500;
  min-width: 90px !important;
}
[data-baseweb="tab"]:hover {
  color: #FFE44D !important;
}

/* ============================================================== 
   ⚙️ UNIVERSAL WIDGET ALIGNMENT (FORMS / OCR / SETTINGS)
   ============================================================== */
.stTextInput, .stNumberInput, .stTextArea, .stSelectbox,
.stSlider, .stCheckbox, .stFileUploader {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 6px auto !important;
}
.stTextInput > div > div > input,
.stTextArea > div > textarea,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
  background: #1A1A1A !important;
  color: var(--light-text) !important;
  border: 1px solid #555 !important;
  border-radius: 6px !important;
  width: 260px !important;
  padding: 6px 10px !important;
  margin: 6px auto !important;
  display: block !important;
  text-align: left !important;
}
.stCheckbox > label {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  color: var(--gold) !important;
}
label {
  text-align: left !important;
  width: 260px !important;
  color: var(--gold) !important;
  margin-bottom: 4px !important;
}
.stSlider { width: 260px !important; margin: 6px auto !important; }

/* Dropdown background */
.stSelectbox > div > div > div[data-baseweb="select"] {
  background: #1A1A1A !important;
  border: 1px solid #555 !important;
  border-radius: 6px !important;
  width: 260px !important;
}

/* ============================================================== 
   🟨 BUTTONS (Login, Save, etc.)
   ============================================================== */
.stButton > button {
  background: linear-gradient(90deg, #FFD700, #E6BE00);
  color: black !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  width: 240px !important;
  margin: 10px auto !important;
  transition: all 0.3s ease-in-out !important;
  border: none !important;
}
.stButton > button:hover {
  background: linear-gradient(90deg, #FFE44D, #FFD700);
  transform: scale(1.03);
}

/* ============================================================== 
   ✨ INPUT FOCUS EFFECT
   ============================================================== */
.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stTextArea > div > textarea:focus {
  outline: none !important;
  border: 1px solid var(--gold) !important;
  box-shadow: 0 0 8px var(--gold) !important;
}

/* ============================================================== 
   🟡 SLIDER GOLDEN THEME
   ============================================================== */
.stSlider > div > div > div[data-baseweb="slider"] > div {
  background: var(--gold) !important;
}
.stSlider > div [role="slider"] {
  background: var(--gold) !important;
  border: 2px solid white !important;
  box-shadow: 0 0 6px rgba(255,215,0,0.7) !important;
}

</style>
""", unsafe_allow_html=True)


# ---------- your form goes here ----------
# with st.form("new_thread_form", clear_on_submit=True):
#     new_name = st.text_input("New conversation name", value="")
#     create = st.form_submit_button("Create")
# st.markdown("</div>", unsafe_allow_html=True)  # <-- CLOSE the wrapper right after the form





# --- i18n ---
I18N = {
    "en": {
        "brand_tag":"We find. Smarter.",
        "nav_dashboard":"Dashboard","nav_chat":"Conversational Recruiter","nav_explorer":"Explorer","nav_assess": "Assessment & Insights", "nav_customfactors": "Fit Metrics", "nav_bias":"Bias & Explainability","nav_settings":"Settings", "nav_vacancies":"Live Vacancies Portal", "nav_documentation": "Functional Documentation",
        "title_dashboard":"CynthAI© TalentLens Dashboard","title_chat":"CynthAI© Conversational Recruiter","title_explorer":"Candidate Explorer","title_bias":"Bias & Explainability","title_settings":"Settings & Data",
        "kpi_total":"Total Candidates","kpi_avg_success":"Avg. Success Probability","kpi_avg_fit":"Avg. Culture Fit","dist_motivation":"Distribution of Motivation Scores",
        "lang_label":"Language","api_key":"Enter your OpenAI API key","logo_tip":"Ensure 'CynthAI_Logo.png' is present for the sidebar logo.",
        "upload_cv":"Upload Candidate CV (PDF/DOCX/TXT)","upload_cl":"Upload Cover Letter (optional)","paste_cl":"Paste Cover Letter Text (optional)",
        "select_sector":"Sector","select_vac":"Select Vacancy","or_upload_vac":"Or upload a vacancy CSV",
        "pred_prob":"Predicted Success Probability","motivation":"Motivation","skillmatch":"Skill Match","culturefit":"Culture Fit","sentiment":"Sentiment","exp_years":"Experience (Years)",
        "what_if":"What-if Simulator","salary_boost":"Salary increase (%)","remote_days":"Remote days/week","offer_uplift":"Adjusted Probability (est.)",
        "chat_hint":"Ask me to analyze the uploaded CV for the selected vacancy (EN or NL).","shap_title":"Feature Importance (SHAP)","fairness":"Fairness slice (by Gender)","sample_loaded":"Loaded sector catalog and dataset.",
        "msg_outreach":"Generate Personalized Outreach","candidate_name":"Candidate name","role_title":"Role title","generate":"Generate","copy_hint":"Copy and tweak before sending."
    },
    "nl": {
        "brand_tag":"Wij vinden. Slimmer.",
        "nav_dashboard":"Dashboard","nav_chat":"Conversational Recruiter","nav_explorer":"Verkenner","nav_assess": "Beoordeling & Inzichten","nav_customfactors": "Fit Metrics","nav_bias":"Bias & Uitlegbaarheid","nav_settings":"Instellingen","nav_vacancies":"Live Vacancies Portal",  "nav_documentation": "Functionele Documentatie",
        "title_dashboard":"CynthAI© TalentLens Dashboard","title_chat":"CynthAI© Conversational Recruiter","title_explorer":"Kandidaten Verkenner","title_bias":"Bias & Uitlegbaarheid","title_settings":"Instellingen & Data",
        "kpi_total":"Totaal kandidaten","kpi_avg_success":"Gem. succeskans","kpi_avg_fit":"Gem. cultuurfit","dist_motivation":"Verdeling Motivatiescores",
        "lang_label":"Taal","api_key":"Vul je OpenAI API-sleutel in","logo_tip":"Zorg dat 'CynthAI_Logo.png' aanwezig is voor het zijbalklogo.",
        "upload_cv":"Upload CV kandidaat (PDF/DOCX/TXT)","upload_cl":"Upload Motivatiebrief (optioneel)","paste_cl":"Plak tekst motivatiebrief (optioneel)",
        "select_sector":"Sector","select_vac":"Kies vacature","or_upload_vac":"Of upload een vacature-CSV",
        "pred_prob":"Voorspelde succeskans","motivation":"Motivatie","skillmatch":"Skill Match","culturefit":"Cultuurfit","sentiment":"Sentiment","exp_years":"Ervaring (jaren)",
        "what_if":"Wat-als Simulator","salary_boost":"Salarisverhoging (%)","remote_days":"Dagen thuiswerk/week","offer_uplift":"Aangepaste kans (schatting)",
        "chat_hint":"Vraag mij om het geüploade CV te analyseren voor de gekozen vacature (NL of EN).","shap_title":"Belangrijkste kenmerken (SHAP)","fairness":"Fairness per Gender","sample_loaded":"Sectorcatalogus en dataset geladen.",
        "msg_outreach":"Genereer Persoonlijk Bericht","candidate_name":"Naam kandidaat","role_title":"Functietitel","generate":"Genereer","copy_hint":"Kopieer en verfijn voor verzending."
    }
}
def t(key, lang="en"): return I18N.get(lang, I18N["en"]).get(key, key)

# --- Data: sectors & vacancies ---
def sample_vacancies_by_sector():
    return {
        "IT":[{"JobTitle":"Data Analyst","RequiredSkills":["Python","SQL","PowerBI","Visualization","Statistics"],"ValueWords":["analysis","autonomy","curiosity","impact","learning"],"ExpMin":2,"ExpMax":6},
              {"JobTitle":"Data Engineer","RequiredSkills":["Python","SQL","ETL","Airflow","Cloud"],"ValueWords":["ownership","craft","quality","scalability","learning"],"ExpMin":3,"ExpMax":8},
              {"JobTitle":"Software Developer","RequiredSkills":["Python","JavaScript","Git","APIs","Testing"],"ValueWords":["craft","innovation","autonomy","teamwork","impact"],"ExpMin":2,"ExpMax":7}],
        "HR":[{"JobTitle":"HR Consultant","RequiredSkills":["Stakeholder","Advisory","Recruitment","Policy","Communication"],"ValueWords":["empathy","collaboration","trust","structure","clarity"],"ExpMin":3,"ExpMax":8},
              {"JobTitle":"Recruiter","RequiredSkills":["Sourcing","Screening","Interviewing","ATS","EmployerBranding"],"ValueWords":["connection","clarity","speed","quality","partnership"],"ExpMin":1,"ExpMax":5}],
        "Marketing":[{"JobTitle":"Marketing Manager","RequiredSkills":["Campaigns","Brand","SEO","Content","Leadership"],"ValueWords":["creativity","ownership","innovation","storytelling","growth"],"ExpMin":4,"ExpMax":10},
                     {"JobTitle":"Content Marketer","RequiredSkills":["Copywriting","SEO","Analytics","Social","CMS"],"ValueWords":["storytelling","clarity","growth","curiosity","impact"],"ExpMin":1,"ExpMax":5}],
        "Logistics":[{"JobTitle":"Logistics Planner","RequiredSkills":["Planning","WMS","Excel","Communication","Problem-solving"],"ValueWords":["structure","ownership","reliability","teamwork","service"],"ExpMin":1,"ExpMax":6},
                     {"JobTitle":"Supply Chain Analyst","RequiredSkills":["SQL","Forecasting","PowerBI","ERP","Inventory"],"ValueWords":["analysis","precision","improvement","collaboration","impact"],"ExpMin":2,"ExpMax":7}],
        "Finance":[{"JobTitle":"Financial Controller","RequiredSkills":["Accounting","Excel","Reporting","IFRS","Analysis"],"ValueWords":["accuracy","integrity","ownership","clarity","structure"],"ExpMin":3,"ExpMax":9},
                   {"JobTitle":"Business Analyst","RequiredSkills":["Modelling","SQL","PowerBI","Stakeholder","Budgeting"],"ValueWords":["impact","analysis","learning","partnership","quality"],"ExpMin":2,"ExpMax":7}],
        "Sales":[{"JobTitle":"Account Manager","RequiredSkills":["Prospecting","Negotiation","CRM","Forecasting","Presentation"],"ValueWords":["ownership","growth","relationship","drive","results"],"ExpMin":2,"ExpMax":8}],
        "Engineering":[{"JobTitle":"Mechanical Engineer","RequiredSkills":["CAD","FEA","Materials","Testing","Manufacturing"],"ValueWords":["craft","precision","innovation","safety","quality"],"ExpMin":2,"ExpMax":8}],
        "Legal":[{"JobTitle":"Legal Counsel","RequiredSkills":["Contract","Compliance","GDPR","Negotiation","Advisory"],"ValueWords":["integrity","precision","clarity","risk","trust"],"ExpMin":3,"ExpMax":9}],
        "Healthcare":[{"JobTitle":"Healthcare Administrator","RequiredSkills":["EMR","Scheduling","Compliance","Communication","Billing"],"ValueWords":["care","trust","structure","service","quality"],"ExpMin":1,"ExpMax":6}]
    }
def default_sector(): return "IT"
SECTORS = sample_vacancies_by_sector()

# --- Helpers for vacancy extraction / OCR ---
def _tess_lang_code(ui_lang: str, override_label: str) -> str:
    if override_label.startswith("English"): return "eng"
    if override_label.startswith("Dutch"): return "nld"
    return "nld" if str(ui_lang).lower().startswith("nl") else "eng"

def _tess_psm_value(psm_label: str) -> int:
    try:
        return int(psm_label.split(" - ", 1)[0].strip())
    except Exception:
        return 3

def _preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    try:
        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray)
        return gray.point(lambda p: 255 if p > 180 else 0)
    except Exception:
        return img

def _ocr_pdf_bytes(pdf_bytes: bytes,
                   ui_lang: str = "en",
                   override_lang_label: str = "Auto (based on UI language)",
                   max_pages: int = 5,
                   poppler_bin: str = "",
                   dpi: int = 300,
                   psm_label: str = "3 - Fully auto") -> str:
    if not OCR_AVAILABLE or not pdf_bytes:
        return ""
    try:
        kwargs = dict(fmt="png", first_page=1, last_page=max_pages, dpi=int(dpi))
        if poppler_bin:
            kwargs["poppler_path"] = poppler_bin
        images = convert_from_bytes(pdf_bytes, **kwargs)
        tess_lang = _tess_lang_code(st.session_state.get("lang_hint","en"), override_lang_label)
        psm = _tess_psm_value(psm_label)
        config = f"--psm {psm}"
        texts = []
        for img in images:
            base = _preprocess_for_ocr(img)
            cand_texts = []
            for rot in (0, 90, 270):
                try:
                    img_rot = base if rot == 0 else base.rotate(rot, expand=True)
                    cand_texts.append(pytesseract.image_to_string(img_rot, lang=tess_lang, config=config))
                except Exception:
                    cand_texts.append("")
            page_text = max(cand_texts, key=lambda s: len(s or ""))
            texts.append(page_text)
        return "\n".join(texts)
    except Exception:
        return ""

def extract_text_from_docx(file):
    if docx is None: return ""
    try: return "\n".join(p.text for p in docx.Document(file).paragraphs)
    except Exception: return ""

def extract_text_from_pdf(uploaded_file):
    if uploaded_file is None: return ""
    # Bytes buffer
    try:
        data = uploaded_file.getvalue()
    except Exception:
        try: data = uploaded_file.read()
        except Exception: data = b""
    text = ""

    # A) pdfminer
    if pdf_extract_text is not None and data:
        try: text = pdf_extract_text(io.BytesIO(data)) or ""
        except Exception: text = ""

    # B) PyPDF fallback
    if (not text or len(text.strip()) < 60) and PYPDF_OK and data:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if getattr(reader, "is_encrypted", False):
                try: reader.decrypt("")
                except Exception: pass
            parts = []
            for page in reader.pages[:15]:
                try: parts.append(page.extract_text() or "")
                except Exception: parts.append("")
            alt = "\n".join(parts).strip()
            if len(alt) > len(text): text = alt
        except Exception:
            pass

    # C) OCR fallback (enhanced)
    if (st.session_state.get("ocr_enabled", False) or len(text.strip()) < 60) and OCR_AVAILABLE and data:
        ocr_text = _ocr_pdf_bytes(
            data,
            ui_lang=st.session_state.get("lang_hint","en"),
            override_lang_label=st.session_state.get("ocr_lang_label","Auto (based on UI language)"),
            max_pages=st.session_state.get("ocr_pages",5),
            poppler_bin=st.session_state.get("poppler_dir","").strip(),
            dpi=st.session_state.get("ocr_dpi",300),
            psm_label=st.session_state.get("ocr_psm","3 - Fully auto")
        )
        if len((ocr_text or "").strip()) > len(text.strip()):
            text = ocr_text

    return text or ""

def extract_cv_text(upload):
    if upload is None: return ""
    name = upload.name.lower()
    if name.endswith(".pdf"): return extract_text_from_pdf(upload)
    if name.endswith(".docx"): return extract_text_from_docx(upload)
    try: return upload.read().decode("utf-8", errors="ignore")
    except Exception: return ""

def extract_texts(uploads):
    """Return combined_text, names_list, display_name for 0..N files (robust)."""
    if not uploads: return "", [], ""
    if not isinstance(uploads, list): uploads = [uploads]
    texts, names = [], []
    for up in uploads:
        txt = extract_cv_text(up)
        fname = getattr(up, "name", "file")
        if txt:
            texts.append(f"\n\n### FILE: {fname}\n{txt}")
            names.append(fname)
    combined = "".join(texts).strip()
    if not names:
        fallback = [getattr(up, "name", "file") for up in uploads]
        display = fallback[0] + (f" (+{len(fallback)-1} more)" if len(fallback) > 1 else "")
        return combined, fallback, display
    display = names[0] + (f" (+{len(names)-1} more)" if len(names) > 1 else "")
    return combined, names, display

# --- NLP feature engineering ---
def basic_clean(text: str): return re.sub(r"\s+", " ", text or "").strip()

import re, datetime, numpy as np

def detect_years_experience(text: str) -> int:
    """Estimate realistic years of experience with contextual filtering."""
    if not text:
        return 0

    t = text.lower()
    yrs = 0

    # --- Context-based pattern (stronger weight when 'experience' nearby) ---
    context_matches = re.findall(
        r"(?:over|about|around|approximately|up to)?\s*(\d{1,2})\s*(?:\+?\s*)?(?:years?|yrs?)\s*(?:of\s+)?(?:work|experience|career)?",
        t,
    )
    if context_matches:
        yrs = max(int(x) for x in context_matches if x.isdigit())

    # --- If no explicit phrase found, try a 'since YEAR' or range heuristic ---
    if yrs == 0:
        # e.g., "since 2014", "2015–2022"
        years = re.findall(r"(19|20)\d{2}", t)
        if len(years) >= 2:
            yrs = max(0, min(int(max(years)) - int(min(years)), 40))
        elif years:
            yrs = max(0, min(datetime.datetime.now().year - int(years[0]), 40))
        else:
            yrs = 0

    # --- Sanity limits ---
    if yrs < 0:
        yrs = 0
    yrs = int(np.clip(yrs, 0, 20))  # limit to 20 years max realistic value
    return yrs

SKILL_VOCAB = {
"Data Analyst":["Python","SQL","PowerBI","Tableau","Statistics","ETL","Pandas","Numpy","Visualization","Dashboards"],
"Data Engineer":["Python","SQL","ETL","Airflow","Cloud","Pandas","Spark"],
"Software Developer":["Python","JavaScript","Git","APIs","Testing","CI/CD"],
"HR Consultant":["Recruitment","Policy","HRIS","Stakeholder","Coaching","Onboarding","Compensation","Benefits","Compliance","Communication"],
"Recruiter":["Sourcing","Screening","Interviewing","ATS","EmployerBranding","LinkedIn"],
"Marketing Manager":["Campaigns","Brand","SEO","SEM","Content","Copywriting","Analytics","Social","Leadership","Strategy"],
"Content Marketer":["Copywriting","SEO","Analytics","Social","CMS","Content"],
"Logistics Planner":["Planning","WMS","Excel","Communication","Problem-solving"],
"Supply Chain Analyst":["SQL","Forecasting","PowerBI","ERP","Inventory"],
"Financial Controller":["Accounting","Excel","Reporting","IFRS","Analysis"],
"Business Analyst":["Modelling","SQL","PowerBI","Stakeholder","Budgeting"],
"Account Manager":["Prospecting","Negotiation","CRM","Forecasting","Presentation"],
"Mechanical Engineer":["CAD","FEA","Materials","Testing","Manufacturing"],
"Legal Counsel":["Contract","Compliance","GDPR","Negotiation","Advisory"],
"Healthcare Administrator":["EMR","Scheduling","Compliance","Communication","Billing"]}

def detect_skills(text: str, role: str) -> float:
    vocab = SKILL_VOCAB.get(role, [])
    if not text or not vocab: return 0.0
    found = sum(1 for kw in vocab if re.search(r"\b"+re.escape(kw)+r"\b", text, re.I))
    return found / len(vocab)

EMOTION_LEX = {
"joy":["happy","delight","enjoy","excited","proud","satisfied","enthusiastic","passion"],
"trust":["trust","reliable","integrity","dependable","responsible","commitment"],
"anticipation":["eager","looking forward","anticipate","expect","curious","aspire","ambition"],
"surprise":["surprise","unexpected","discovery","novel","breakthrough"],
"sadness":["sad","regret","unhappy","disappointed","loss","depress"],
"anger":["angry","frustrated","upset","annoyed","irritated"],
"fear":["afraid","fear","concern","worried","anxious","risk"],
"disgust":["disgust","gross","repulsed","unethical","unfair"]}
def emotion_vector(text: str) -> dict:
    t = (text or "").lower(); vec = {}
    for emo, kws in EMOTION_LEX.items():
        hits = sum(len(re.findall(r"\b"+re.escape(kw)+r"\b", t)) for kw in kws)
        vec[emo] = min(hits/10.0, 1.0)
    return vec

def sentiment_score(text: str) -> float:
    if text is None or len(text)<20: return 0.5
    if TRANSFORMERS_AVAILABLE:
        try:
            pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            res = pipe(text[:1500])[0]; label = res.get("label","NEU").upper(); score = res.get("score",0.5)
            if label.startswith("POS"): return float(np.clip(0.6+0.4*score,0,1))
            if label.startswith("NEG"): return float(np.clip(0.4-0.4*score,0,1))
            return 0.5
        except Exception:
            pass
    pos = len(re.findall(r"\b(excellent|achieved|improved|growth|success|impact|passion|motiv)\w*\b", text.lower()))
    neg = len(re.findall(r"\b(problem|issue|failure|struggle|weak)\w*\b", text.lower()))
    return float(np.clip(0.5+0.03*(pos-neg),0,1))

def culture_fit_score(text: str, value_words: list) -> float:
    if not text or not value_words: return 0.5
    hits = sum(1 for w in value_words if re.search(r"\b"+re.escape(w)+r"\b", text, re.I))
    return hits/max(len(value_words),1)

def education_level_from_text(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\bwo\b|\bmaster\b|\bmsc\b|\buniversity\b|\buniversiteit\b", t): return "WO"
    if re.search(r"\bhbo\b|\bbachelor\b", t): return "HBO"
    if re.search(r"\bmbo\b", t): return "MBO"
    return "HBO"

def build_feature_row(role: str, combined_text: str, vacancy_row: dict, sector: str=None):
    yrs = detect_years_experience(combined_text)
    mot = sentiment_score(combined_text)
    skill = detect_skills(combined_text, role)
    fit = culture_fit_score(combined_text, vacancy_row.get("ValueWords", []))
    emo = emotion_vector(combined_text)
    emo_pos = (emo.get("joy",0)+emo.get("trust",0)+emo.get("anticipation",0)+emo.get("surprise",0))/4.0
    emo_neg = (emo.get("sadness",0)+emo.get("anger",0)+emo.get("fear",0)+emo.get("disgust",0))/4.0
    edu = education_level_from_text(combined_text)
    return {
        "ExperienceYears":yrs,"MotivationScore":mot,"SkillMatch":skill,"CultureFit":fit,"SentimentScore":mot,
        "EmotionPos":float(emo_pos),"EmotionNeg":float(emo_neg),
        "EducationLevel_HBO":1 if edu=="HBO" else 0,"EducationLevel_WO":1 if edu=="WO" else 0
    }

def predict_prob(feat: dict, sector: str=None) -> float:
    row = pd.DataFrame([feat])
    if sector:
        for c in feature_cols:
            if c.startswith("Sector_"): row[c]=0
        sec_col = f"Sector_{sector}"
        if sec_col in feature_cols: row[sec_col]=1
    for c in feature_cols:
        if c not in row.columns: row[c]=0
    row = row[feature_cols]
    return float(model.predict_proba(row)[0][1])

def offer_uplift(base_prob: float, salary_pct: float, remote_days: int) -> float:
    uplift = 0.002*float(salary_pct) + 0.01*min(int(remote_days),3)
    return float(np.clip(base_prob + uplift, 0, 1))

def acceptance_probability(success_prob: float, feat: dict, salary_pct: float=0.0, remote_days: int=0) -> float:
    s = float(success_prob)
    sent = float(feat.get("SentimentScore",0.5))
    emo_pos = float(feat.get("EmotionPos",0.5))
    offer = 0.002*float(salary_pct) + 0.01*min(int(remote_days),3)
    z = 1.2*s + 0.4*sent + 0.4*emo_pos + offer - 0.6
    return float(1/(1+np.exp(-z)))

# --- OpenAI client ---
def get_openai_client(api_key: str):
    if not api_key or OpenAI is None: return None
    try: return OpenAI(api_key=api_key)
    except Exception: return None

# --- Retrieval helpers ---
def _chunk_text(text: str, size: int = 1000, overlap: int = 150):
    text = text or ""
    if len(text) <= size: return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

def build_index(corpus_text: str):
    chunks = _chunk_text(corpus_text, size=1000, overlap=150)
    if not chunks: return {"chunks": [], "tfidf": None}
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(chunks)
    return {"chunks": chunks, "tfidf": (vectorizer, X)}

def retrieve(question: str, index, top_k: int = 8):
    if not question or not index or not index.get("tfidf"): return []
    vectorizer, X = index["tfidf"]
    qv = vectorizer.transform([question])
    sims = cosine_similarity(qv, X).ravel()
    top = sims.argsort()[::-1][:top_k]
    return [(index["chunks"][i], float(sims[i])) for i in top]

# --- LLM narrative + Q&A (grounded) ---
def gpt_narrative_and_qa(client, lang, role, prob, feat_dict, vacancy_row,
                         question=None, corpus_text=None, retrieved=None):
    if client is None:
        if lang == "nl":
            base = (f"Profielschets:\n- Voorspelde succeskans: {prob:.0%}\n- Motivatie: {feat_dict['MotivationScore']:.0%}\n"
                    f"- Skills match: {feat_dict['SkillMatch']:.0%}\n- Cultuurfit: {feat_dict['CultureFit']:.0%}\n"
                    f"- Ervaring: {feat_dict['ExperienceYears']} jaar\n- Rol: {role}\n\n")
            return base + ("(Geen API-sleutel: Q&A/narratief beperkt.)" if question else
                           "Narratief (beperkt zonder model).")
        base = (f"Profile:\n- Predicted success: {prob:.0%}\n- Motivation: {feat_dict['MotivationScore']:.0%}\n"
                f"- Skill match: {feat_dict['SkillMatch']:.0%}\n- Culture fit: {feat_dict['CultureFit']:.0%}\n"
                f"- Experience: {feat_dict['ExperienceYears']} years\n- Role: {role}\n\n")
        return base + ("(No API key: Q&A/narrative limited.)" if question else "Narrative (limited without model).")

    retrieved_text = "\n\n---\n\n".join([c for c,_s in (retrieved or [])]) if retrieved else ""
    mode = "qa" if (question or "").strip() else "narrative"

    system = (
        "You are a bilingual recruitment copilot. "
        "Use only the provided CONTEXT to write your answer. "
        "If the context is insufficient, say so. Keep tone professional, fair, and specific. "
        "If 'lang' == 'nl' respond in Dutch; otherwise English."
    )

    if mode == "narrative":
        user = {
            "lang": lang, "task": "narrative", "role": role, "probability": prob,
            "features": feat_dict, "vacancy": vacancy_row,
            "instructions": ("Write a 6–10 bullet executive narrative of the candidate based on CONTEXT. "
                             "Cover: core profile, key achievements, relevant skills vs. vacancy, culture/value alignment, "
                             "risks/gaps, and recommendation. Ground claims with 'In the CV...' or 'In the letter...' "
                             "when appropriate."),
            "CONTEXT": (corpus_text or "")[:150000]
        }
    else:
        user = {
            "lang": lang, "task": "qa", "role": role, "question": question,
            "instructions": ("Answer strictly from the retrieved snippets below. "
                             "If uncertain, say you can't find it. Use concise bullets when helpful."),
            "RETRIEVED_SNIPPETS": retrieved_text[:150000]
        }

    try:
        out = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(user)}],
            temperature=0.4, max_tokens=700
        )
        return out.choices[0].message.content
    except Exception:
        return gpt_narrative_and_qa(None, lang, role, prob, feat_dict, vacancy_row, question, corpus_text, retrieved)

# --- Batch narrative helper (per-CV) ---
def generate_narrative_for_single_cv(
    upload_file,
    role: str,
    vac_row: dict,
    vacancy_txt: str,
    lang: str,
    sector_for_model: str,
    openai_key: str,
    cover_letter_text: str = ""
):
    """Return dict with filename, narrative, and key metrics for one CV file."""
    # 1) Extract text for this specific CV
    cv_text = extract_cv_text(upload_file)
    fname = getattr(upload_file, "name", "candidate")

    # 2) Build per-CV corpus (vacancy + this CV + optional cover letter)
    vacancy_section = f"\n\n### VACANCY MATCHED CONTEXT\n{vacancy_txt}\n" if vacancy_txt else ""
    cl_section = f"\n\n### COVER LETTER\n{cover_letter_text}\n" if cover_letter_text else ""
    corpus_text = basic_clean((vacancy_section + f"\n\n### FILE: {fname}\n{cv_text}" + cl_section).strip())

    # 3) Features & probabilities
    feat = build_feature_row(role, corpus_text, vac_row, sector=sector_for_model)
    base_prob = predict_prob(feat, sector=sector_for_model)
    _w = get_sector_weights(sector_for_model)
    _blend = st.session_state.get("blend", 0.4)
    adj_prob = adjust_with_custom_factors(base_prob, feat, _w, _blend)

    # 4) Narrative via LLM (or lightweight fallback)
    client = get_openai_client(openai_key)
    narrative = gpt_narrative_and_qa(
        client, lang, role, adj_prob, feat, vac_row,
        question=None, corpus_text=corpus_text, retrieved=None
    )

    # 5) Collect outputs
    out = {
        "filename": fname,
        "role": role or "-",
        "pred_success_adj": adj_prob,
        "pred_success_model": base_prob,
        "skill_match": float(feat.get("SkillMatch", 0.0)),
        "culture_fit": float(feat.get("CultureFit", 0.0)),
        "motivation": float(feat.get("MotivationScore", 0.0)),
        "experience_years": int(feat.get("ExperienceYears", 0)),
        "narrative": narrative.strip()
    }
    return out

# --- Custom Factors ---
def get_sector_weights(sector: str):
    defaults = {"ExperienceYears":0.6,"SkillMatch":1.4,"CultureFit":1.2,"MotivationScore":1.0,"SentimentScore":0.8,"EmotionPos":0.6,"EmotionNeg":0.4}
    if "weights" not in st.session_state: st.session_state["weights"] = {}
    if sector not in st.session_state["weights"]: st.session_state["weights"][sector] = defaults.copy()
    return st.session_state["weights"][sector]

def adjust_with_custom_factors(base_prob: float, feat: dict, weights: dict, blend: float=0.4) -> float:
    norm = {"ExperienceYears": min(max(float(feat.get("ExperienceYears",0))/10.0,0.0),1.0)}
    for k in ["SkillMatch","CultureFit","MotivationScore","SentimentScore","EmotionPos","EmotionNeg"]:
        v = feat.get(k,0.0); norm[k] = 0.0 if v is None else max(0.0, min(1.0, float(v)))
    pos = (weights.get("ExperienceYears",0)*norm["ExperienceYears"] +
           weights.get("SkillMatch",0)*norm["SkillMatch"] +
           weights.get("CultureFit",0)*norm["CultureFit"] +
           weights.get("MotivationScore",0)*norm["MotivationScore"] +
           weights.get("SentimentScore",0)*norm["SentimentScore"] +
           weights.get("EmotionPos",0)*norm["EmotionPos"])
    neg = weights.get("EmotionNeg",0)*norm["EmotionNeg"]
    raw = pos - 0.7*neg
    pos_sum = sum([weights.get(k,0) for k in ["ExperienceYears","SkillMatch","CultureFit","MotivationScore","SentimentScore","EmotionPos"]]) or 1.0
    score = max(0.0, min(1.0, raw/pos_sum))
    blend = max(0.0, min(1.0, float(blend)))
    return float(max(0.0, min(1.0, (1.0-blend)*base_prob + blend*score)))

# ---------------- Sidebar (incl. NEW vacancy pipeline; safe state) ----------------
with st.sidebar:
    # Always-available GitHub raw logo (works locally & on Streamlit Cloud)
    logo_url = "https://raw.githubusercontent.com/dade85/hr_recruitment_update/6e159ada1056008ae6fb7ddf27e94361e19f5881/CynthAI_Logo.png"
    try:
        st.image(logo_url, width=220)
    except Exception:
        # graceful fallback
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/a/ad/Recruitee_logo.png",
            width=180,
        )
        st.caption("CynthAI© TalentLens")

    st.markdown("### 🧭 Navigation")

    # --- Centered Branding Text ---
    st.markdown(
        """
        <div style='text-align:center; font-weight:700; font-size:16px; color:white;'>
            CynthAI© TalentLens
        </div>
        <hr style='border:0.5px solid rgba(255,255,255,0.2); margin:0.8em 0;' />
        """,
        unsafe_allow_html=True
    )

    lang = st.selectbox(t("lang_label","en"), ["nl","en"], index=0, key="lang")
    st.session_state.setdefault("lang_hint", lang)

    # API key (env + input)
    env_default = os.getenv("OPENAI_API_KEY", "")
    default_mask = "••••••••" if env_default else ""
    user_key = st.text_input(t("api_key", lang), type="password", value=default_mask)
    openai_key = user_key.strip() if (user_key and "•" not in user_key) else env_default

    # OCR controls (persisted)
    st.session_state["ocr_enabled"] = st.checkbox(
        "Enable OCR for scanned PDFs", value=st.session_state.get("ocr_enabled", False),
        help="Use for image-only PDFs (requires Tesseract + Poppler)."
    )
    st.session_state["ocr_pages"] = st.slider("OCR pages (first N)", 1, 15, st.session_state.get("ocr_pages", 5), 1)
    st.session_state["poppler_dir"] = st.text_input("Poppler bin path (optional)", value=st.session_state.get("poppler_dir", os.getenv("POPPLER_PATH","")))
    tess_cmd = st.text_input("Tesseract exe path (optional)", value=os.getenv("TESSERACT_CMD",""))
    try:
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
    except Exception:
        pass
    st.caption(f"OCR available: {'Yes' if OCR_AVAILABLE else 'No'}")

    st.session_state["ocr_lang_label"] = st.selectbox(
        "OCR language",
        ["Auto (based on UI language)", "English (eng)", "Dutch (nld)"],
        index=["Auto (based on UI language)", "English (eng)", "Dutch (nld)"].index(
            st.session_state.get("ocr_lang_label","Auto (based on UI language)")
        )
    )
    st.session_state["ocr_dpi"] = st.slider("OCR DPI", 150, 400, st.session_state.get("ocr_dpi", 300), 25)
    st.session_state["ocr_psm"] = st.selectbox(
        "Tesseract PSM (page segmentation mode)",
        ["3 - Fully auto", "4 - Column/variant", "6 - Uniform block", "11 - Sparse text", "12 - Sparse w/ OSD", "13 - Raw line"],
        index=["3 - Fully auto","4 - Column/variant","6 - Uniform block","11 - Sparse text","12 - Sparse w/ OSD","13 - Raw line"].index(
            st.session_state.get("ocr_psm","3 - Fully auto")
        )
    )

    st.markdown("---"); st.caption(t("sample_loaded", lang))

    # ---------- 1) VACANCY AUTO-DETECT (runs BEFORE selectors) ----------
    st.subheader("Vacancy (auto-detect)")
    vac_file = st.file_uploader("Upload Job Vacancy (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="vacancy_file_upl")
    vac_text_paste = st.text_area("Or paste job vacancy text", height=140, key="vacancy_text_paste")

    def _extract_vacancy_text(file):
        if file is None: return ""
        name = file.name.lower()
        if name.endswith(".pdf"): return extract_text_from_pdf(file)
        if name.endswith(".docx"): return extract_text_from_docx(file)
        try: return file.read().decode("utf-8", errors="ignore")
        except Exception: return ""

    def _classify_vacancy_text(vtext: str):
        if not vtext: return None, None, 0.0
        all_vacs = []
        for sec, vacs in SECTORS.items():
            for v in vacs:
                sig = " ".join([v["JobTitle"], " ".join(v.get("RequiredSkills", [])), " ".join(v.get("ValueWords", []))])
                all_vacs.append({"Sector": sec, "JobTitle": v["JobTitle"], "Text": sig})
        df_all = pd.DataFrame(all_vacs)
        tfidf = TfidfVectorizer(stop_words="english")
        X = tfidf.fit_transform(df_all["Text"])
        qv = tfidf.transform([vtext])
        sims = cosine_similarity(qv, X).ravel()
        best_idx = int(np.argmax(sims))
        return df_all.iloc[best_idx]["Sector"], df_all.iloc[best_idx]["JobTitle"], float(sims[best_idx])

    _vac_text = _extract_vacancy_text(vac_file) if vac_file is not None else (vac_text_paste.strip() if vac_text_paste else "")
    if _vac_text:
        best_sector, best_role, score = _classify_vacancy_text(_vac_text)
        if best_sector and best_role:
            changed = (
                st.session_state.get("sector", default_sector()) != best_sector or
                st.session_state.get("vacancy_select") != best_role
            )
            st.session_state["auto_vac_text"] = _vac_text
            st.session_state["auto_vac_score"] = score
            if changed:
                st.session_state["sector"] = best_sector
                st.session_state["vacancy_select"] = best_role
                st.toast(f"Vacancy matched → Sector: {best_sector} · Role: {best_role}", icon="✅")
                st.rerun()
            else:
                st.caption(f"Detected sector: {best_sector} · role: {best_role} · similarity {score:.3f}")
        else:
            st.warning("Could not confidently classify the vacancy. You can still select sector/role manually.")
    else:
        st.caption("Upload or paste a vacancy to auto-detect sector and role.")
    # --------------------------------------------------------------------

    # ---------- 2) SELECTORS (widget keys are canonical; no manual post-assignments) ----------
    sector_opts = list(SECTORS.keys())
    sector_default = st.session_state.get("sector", default_sector())
    st.selectbox(
        t("select_sector", lang),
        sector_opts,
        index=sector_opts.index(sector_default) if sector_default in sector_opts else sector_opts.index(default_sector()),
        key="sector"   # widget manages st.session_state["sector"]
    )

    vac_options_for_sidebar = [v["JobTitle"] for v in SECTORS[st.session_state["sector"]]]
    vac_default = st.session_state.get("vacancy_select", (vac_options_for_sidebar[0] if vac_options_for_sidebar else None))
    st.selectbox(
        t("select_vac", lang),
        vac_options_for_sidebar,
        index=vac_options_for_sidebar.index(vac_default) if (vac_default in vac_options_for_sidebar) else 0,
        key="vacancy_select"   # widget manages st.session_state["vacancy_select"]
    )

    # Optional vacancy CSV to swap catalog entries per sector
    vac_upload = st.file_uploader(t("or_upload_vac", lang), type=["csv"])

def get_current_sector(): return st.session_state.get("sector", default_sector())

# --- Synthetic dataset & model ---
@st.cache_data
def make_data(n=1000, seed=13):
    rng = np.random.default_rng(seed)
    edu = rng.choice(["MBO","HBO","WO"], size=n, p=[0.35,0.45,0.20])
    yrs = rng.integers(0, 16, size=n)
    mot = np.clip(rng.normal(0.68,0.15,size=n),0,1)
    skill = np.clip(rng.normal(0.72,0.12,size=n),0.2,1)
    fit = np.clip(rng.normal(0.66,0.18,size=n),0,1)
    sent = np.clip(rng.normal(0.65,0.2,size=n),0,1)
    emo_pos = np.clip(sent + rng.normal(0,0.1,size=n), 0, 1)
    emo_neg = np.clip(1 - sent + rng.normal(0,0.1,size=n), 0, 1)
    gender = rng.choice(["F","M","X"], size=n, p=[0.48,0.48,0.04])
    sector = rng.choice(list(SECTORS.keys()), size=n)
    logit = -1.1 + 0.06*yrs + 0.9*mot + 1.2*skill + 0.9*fit + 0.4*sent + np.where(edu=="WO",0.25,np.where(edu=="HBO",0.15,0)) + rng.normal(0,0.55,size=n)
    prob = 1/(1+np.exp(-logit))
    hired = (prob>0.6).astype(int)
    ret = (1/(1+np.exp(-0.5 + 0.04*yrs + 1.0*fit + 0.6*mot + rng.normal(0,0.5,size=n)))>0.55).astype(int)
    df = pd.DataFrame({
        "Sector":sector,"EducationLevel":edu,"ExperienceYears":yrs,"MotivationScore":mot,"SkillMatch":skill,"CultureFit":fit,
        "SentimentScore":sent,"EmotionPos":emo_pos,"EmotionNeg":emo_neg,"Gender":gender,"Hired":hired,"Retained12m":ret
    })
    return df

df = make_data()

@st.cache_resource
def train_model(df: pd.DataFrame):
    base = df.drop(columns=[c for c in ["Retained12m","Gender"] if c in df.columns])
    work = pd.get_dummies(base, columns=[c for c in ["EducationLevel","Sector"] if c in base.columns], drop_first=True)
    X, y = work.drop(columns=["Hired"]), work["Hired"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:,1]; pred = (prob>0.5).astype(int)
    metrics = {"f1": float(f1_score(yte, pred)), "auc": float(roc_auc_score(yte, prob))}
    return model, X.columns.tolist(), metrics

model, feature_cols, metrics = train_model(df)



# --- Vacancy CSV (optional, per selected sector) ---
def get_vac_df(vac_upload_file, sector=None):
    default_df = pd.DataFrame(SECTORS.get(sector or default_sector(), []))
    if vac_upload_file is None: return default_df.copy()
    try:
        dfv = pd.read_csv(vac_upload_file)
        def split_or(v): return [x.strip() for x in v.split(",") if x.strip()] if isinstance(v,str) else []
        cols = {c.lower(): c for c in dfv.columns}
        mapping = {"jobtitle":"JobTitle","requiredskills":"RequiredSkills","valuewords":"ValueWords","expmin":"ExpMin","expmax":"ExpMax"}
        for low, proper in mapping.items():
            if low in cols: dfv.rename(columns={cols[low]: proper}, inplace=True)
        dfv["RequiredSkills"] = dfv["RequiredSkills"].apply(split_or)
        dfv["ValueWords"] = dfv["ValueWords"].apply(split_or)
        return dfv
    except Exception:
        st.warning("Could not read vacancy CSV; using sector defaults.")
        return default_df.copy()

current_sector = get_current_sector()
vac_df = get_vac_df(vac_upload, current_sector)

# ---------------- TABS ----------------
tabs = [
    f"📊 {t('nav_dashboard', lang)}",
    f"💬 {t('nav_chat', lang)}",
    f"🧭 {t('nav_explorer', lang)}",
    "🧠 Assessment & Insights",
    "⚙️ Custom Factors",  # independent tab linked with assessment
    f"🔍 {t('nav_bias', lang)}",
    f"⚙️ {t('nav_settings', lang)}",
    f"🌐  {t('nav_vacancies', lang)}",
    f"📘 {t('nav_documentation', lang)}"
]
section = st.tabs(tabs)

current_sector = get_current_sector()
vac_df = get_vac_df(vac_upload, current_sector)

# ===== DASHBOARD =====
with section[0]:
    st.title(t("title_dashboard", lang))

    # ---- Matched Vacancy Badge (auto/ manual + similarity score) ----
    badge_sector = st.session_state.get("sector", default_sector())
    badge_role = st.session_state.get("vacancy_select", None)
    badge_auto = bool(st.session_state.get("auto_vac_text"))
    badge_score = st.session_state.get("auto_vac_score", None)
    badge_src = "Auto-detected" if badge_auto else "Manual"
    score_str = f" · similarity {badge_score:.3f}" if (badge_score is not None and badge_auto) else ""
    st.markdown(
        f"""
        <div style="
            display:flex; align-items:center; gap:10px; margin:.25rem 0 1rem 0;
            background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.06);
            border-radius: 999px; padding: 6px 12px; width: fit-content;
        ">
            <span style="
                background: {ACCENT_RED}; color: #fff; font-weight: 800; font-size: 12px;
                padding: 3px 10px; border-radius: 999px; letter-spacing:.2px;
            ">
                Matched vacancy
            </span>
            <span style="color:#fff; opacity:.95; font-size: 13px;">
                Sector: <strong>{badge_sector}</strong> · Role: <strong>{badge_role or "-"}</strong>
                <span style="opacity:.8;">({badge_src}{score_str})</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    # ---------------------------------------------------------------


    # ---------------------------------------------------------------

    c1,c2,c3 = st.columns(3)
    c1.metric(t("kpi_total", lang), len(df))
    c2.metric(t("kpi_avg_success", lang), f"{df['Hired'].mean()*100:.1f}%")
    c3.metric(t("kpi_avg_fit", lang), f"{df['CultureFit'].mean()*100:.1f}%")

    st.markdown("<div class='personato-card'>", unsafe_allow_html=True)
    fig1 = px.histogram(df, x="MotivationScore", nbins=30, title=t("dist_motivation", lang))
    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=TEXT_LIGHT,
        title=dict(font=dict(color=TEXT_LIGHT)), legend=dict(font=dict(color=TEXT_LIGHT)),
        xaxis=dict(title_font=dict(color=TEXT_LIGHT), tickfont=dict(color=TEXT_LIGHT)),
        yaxis=dict(title_font=dict(color=TEXT_LIGHT), tickfont=dict(color=TEXT_LIGHT)),
    )
    fig1.update_traces(marker_color=ACCENT_RED)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Model metrics"):
        m1,m2 = st.columns(2)
        m1.metric("F1", f"{metrics['f1']:.3f}")
        m2.metric("ROC AUC", f"{metrics['auc']:.3f}")

    # Candidate card (after Chat analysis)
    if 'last_features' in st.session_state:
        lf = st.session_state['last_features']; p = st.session_state.get('last_probability',0.0)
        _w = get_sector_weights(current_sector); _blend = st.session_state.get('blend',0.4)
        p_adj = adjust_with_custom_factors(p, lf, _w, _blend)

        st.markdown(f"### Candidate: **{st.session_state.get('last_candidate_name','Candidate')}**  ·  Role: **{st.session_state.get('last_role','-')}**  ·  File: _{st.session_state.get('last_cv_filename','')}_")

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Predicted Success", f"{p_adj*100:.1f}%")
        k2.metric("Acceptance (est.)", f"{acceptance_probability(p_adj, lf)*100:.1f}%")
        k3.metric("Skill Match", f"{lf.get('SkillMatch',0)*100:.0f}%")
        k4.metric("Culture Fit", f"{lf.get('CultureFit',0)*100:.0f}%")
        k5.metric("Motivation", f"{lf.get('MotivationScore',0)*100:.0f}%")

        # Feature Snapshot chart
        snapshot = {
            "Experience": min(lf.get("ExperienceYears",0)/10.0, 1.0),
            "Skill": lf.get("SkillMatch",0),
            "Culture": lf.get("CultureFit",0),
            "Motivation": lf.get("MotivationScore",0),
            "Sentiment": lf.get("SentimentScore",0),
            "Emotion+": lf.get("EmotionPos",0),
            "Emotion-": lf.get("EmotionNeg",0)
        }
        snap_df = pd.DataFrame({"Feature": list(snapshot.keys()), "Score": [snapshot[k] for k in snapshot]})
        fig_snap = px.bar(snap_df, x="Feature", y="Score", title="Feature Snapshot (0–1)")
        fig_snap.update_traces(marker_color=ACCENT_RED)
        fig_snap.update_layout(
            yaxis=dict(range=[0,1], title=""), xaxis=dict(title=""),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=TEXT_LIGHT, title=dict(font=dict(color=TEXT_LIGHT)),
        )
        st.plotly_chart(fig_snap, use_container_width=True)

# ===== CONVERSATIONAL RECRUITER =====
with section[1]:
    st.title(t("title_chat", lang))

    # Conversation store
    if "threads" not in st.session_state:
        st.session_state.threads = {}
    if "current_thread" not in st.session_state:
        st.session_state.current_thread = None

    left, right = st.columns([1, 1])

    # ---------- LEFT COLUMN ----------
    with left:
        # Batch uploads
        cv_files = st.file_uploader(t("upload_cv", lang), type=["pdf", "docx", "txt"], key="cv_file", accept_multiple_files=True)
        cl_files = st.file_uploader(t("upload_cl", lang), type=["pdf", "docx", "txt"], key="cl_file", accept_multiple_files=True)
        cl_text_area = st.text_area(t("paste_cl", lang), height=140, key="cover_text")

        # Vacancy select reflects sidebar
        vac_options_chat = vac_df["JobTitle"].tolist() if not vac_df.empty else []
        default_role = st.session_state.get("vacancy_select", (vac_options_chat[0] if vac_options_chat else None))
        role = st.selectbox(
            t("select_vac", lang),
            vac_options_chat,
            index=vac_options_chat.index(default_role) if default_role in vac_options_chat else (0 if vac_options_chat else None),
            key="chat_role_select"
        )

        # Build corpus from uploads
        cv_combined, cv_names, cv_display = extract_texts(cv_files)
        cl_combined, cl_names, _ = extract_texts(cl_files)
        if not cl_combined and cl_text_area:
            cl_combined = basic_clean(cl_text_area)

        vacancy_txt = st.session_state.get("auto_vac_text", "")
        vacancy_section = f"\n\n### VACANCY MATCHED CONTEXT\n{vacancy_txt}\n" if vacancy_txt else ""

        current_corpus = basic_clean((
            vacancy_section +
            (cv_combined or "") +
            (("\n\n### COVER LETTER\n" + cl_combined) if cl_combined else "")
        ).strip())

        # Warn if OCR required
        if (cv_files or cl_files) and not current_corpus:
            msg = "Uploaded files were read but no text could be extracted. "
            if OCR_AVAILABLE:
                msg += "Try enabling **OCR for scanned PDFs** in the sidebar."
            else:
                msg += "Install Tesseract + Poppler and enable OCR."
            st.warning(msg)

    # ---------- RIGHT COLUMN ----------
    with right:
        st.caption("Manage conversations (threads).")

        # Ensure at least one thread exists
        if not st.session_state.threads:
            tid = str(uuid.uuid4())[:8]
            st.session_state.threads[tid] = {
                "name": "Conversation 1",
                "archived": False,
                "messages": [],
                "corpus": "",
                "index": None,
                "role": None,
                "vac_row": None,
                "display_name": "",
                "vacancy_text": ""
            }
            st.session_state.current_thread = tid

        threads = st.session_state.threads
        active_ids = [tid for tid, meta in threads.items() if not meta.get("archived")]
        if not active_ids:
            any_id = next(iter(threads))
            threads[any_id]["archived"] = False
            active_ids = [any_id]

        def _label(tid):
            meta = threads[tid]
            return f"{meta.get('name', 'Conversation')} · {meta.get('display_name', 'No files')}"

        current_choice = st.selectbox(
            "Select conversation",
            options=active_ids,
            index=max(0, active_ids.index(st.session_state.current_thread))
            if st.session_state.current_thread in active_ids else 0,
            format_func=_label
        )
        st.session_state.current_thread = current_choice
        current_meta = threads[current_choice]

        # New conversation form
        with st.form("new_thread_form", clear_on_submit=True):
            new_name = st.text_input("New conversation name", value="")
            create = st.form_submit_button("Create")
        if create and new_name.strip():
            tid = str(uuid.uuid4())[:8]
            threads[tid] = {
                "name": new_name.strip(),
                "archived": False,
                "messages": [],
                "corpus": "",
                "index": None,
                "role": None,
                "vac_row": None,
                "display_name": "",
                "vacancy_text": ""
            }
            st.session_state.current_thread = tid
            current_meta = threads[tid]

        # Buttons
        cA, cB, cC = st.columns(3)
        with cA:
            if st.button("Attach current files to thread"):
                current_meta["corpus"] = current_corpus
                current_meta["index"] = build_index(current_corpus) if current_corpus else None
                current_meta["role"] = role
                current_meta["vac_row"] = (vac_df[vac_df["JobTitle"] == role].iloc[0].to_dict()
                                           if (role and not vac_df.empty) else {})
                current_meta["display_name"] = cv_display or (cv_names[0] if cv_names else "")
                current_meta["vacancy_text"] = vacancy_txt
                st.success("Files (incl. vacancy context) attached.")
        with cB:
            if not current_meta.get("archived"):
                if st.button("Archive"):
                    current_meta["archived"] = True
                    st.success("Conversation archived.")
            else:
                if st.button("Unarchive"):
                    current_meta["archived"] = False
                    st.success("Conversation unarchived.")
        with cC:
            if st.button("Delete"):
                threads.pop(current_choice, None)
                remaining = [tid for tid, meta in threads.items() if not meta.get("archived")]
                st.session_state.current_thread = remaining[0] if remaining else None
                st.info("Conversation deleted.")
                st.stop()

        st.markdown("---")

        # Render chat
        for speaker, msg in current_meta.get("messages", []):
            with st.chat_message(speaker):
                st.markdown(msg)

        # Chat input
        prompt = st.chat_input("Ask about the attached files, or type anything…")
        if prompt is not None:
            current_meta.setdefault("messages", []).append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            chosen_role = current_meta.get("role") or role
            vac_row = current_meta.get("vac_row") or (vac_df[vac_df["JobTitle"] == chosen_role].iloc[0].to_dict()
                                                      if (chosen_role and not vac_df.empty) else {})
            corpus_text = current_meta.get("corpus", current_corpus)
            idx = current_meta.get("index") or (build_index(corpus_text) if corpus_text else None)
            current_meta["index"] = idx

            # Feature engineering
            feat = (build_feature_row(chosen_role, corpus_text, vac_row, sector=current_sector)
                    if corpus_text else {"ExperienceYears": 0, "MotivationScore": 0.5, "SkillMatch": 0.0,
                                         "CultureFit": 0.5, "SentimentScore": 0.5, "EmotionPos": 0.5,
                                         "EmotionNeg": 0.5})
            base_prob = predict_prob(feat, sector=current_sector)
            _w = get_sector_weights(current_sector)
            _blend = st.session_state.get('blend', 0.4)
            adj_prob = adjust_with_custom_factors(base_prob, feat, _w, _blend)
            acc = acceptance_probability(adj_prob, feat)

            top_chunks = retrieve(prompt, idx, top_k=8) if (prompt and idx) else []
            client = get_openai_client(openai_key)
            answer = gpt_narrative_and_qa(client, lang, chosen_role, adj_prob, feat, vac_row,
                                          question=prompt, corpus_text=corpus_text, retrieved=top_chunks)

            reply = (f"{answer}\n\n---\n**{t('pred_prob', lang)}:** {adj_prob:.0%} "
                     f"(model: {base_prob:.0%}) · **Acceptance (est.)**: {acc:.0%}\n"
                     f"- {t('motivation', lang)}: {feat['MotivationScore']:.0%}\n"
                     f"- {t('skillmatch', lang)}: {feat['SkillMatch']:.0%}\n"
                     f"- {t('culturefit', lang)}: {feat['CultureFit']:.0%}\n"
                     f"- {t('sentiment', lang)}: {feat['SentimentScore']:.0%}\n"
                     f"- {t('exp_years', lang)}: {feat['ExperienceYears']}\n")

            with st.chat_message("assistant"):
                st.markdown(reply)
            current_meta["messages"].append(("assistant", reply))

            # ---- SYNC NARRATIVE TO ASSESSMENT ----
            candidate_id = current_meta.get("display_name") or (cv_names[0] if cv_names else "Candidate")
            narrative_clean = re.sub(r"(\n\s*){2,}", "\n\n", answer.strip())
            st.session_state.setdefault("ai_narratives", {})[candidate_id] = narrative_clean
            current_meta["narrative"] = narrative_clean
            st.info(f"✅ Narrative synced to Assessment for {candidate_id}.")

            # Update dashboard
            st.session_state['last_candidate_name'] = corpus_text.splitlines()[0].strip()[:60] if corpus_text else "Candidate"
            st.session_state['last_features'] = feat
            st.session_state['last_probability'] = base_prob
            st.session_state['last_role'] = chosen_role
            st.session_state['last_sector'] = current_sector
            st.session_state['last_cv_filename'] = current_meta.get("display_name", "")

        # Manual narrative generation
        st.markdown("---")
        if st.button("Generate Narrative from Attached Files"):
            corpus_text = current_meta.get("corpus", current_corpus)
            chosen_role = current_meta.get("role") or role
            vac_row = current_meta.get("vac_row") or (vac_df[vac_df["JobTitle"] == chosen_role].iloc[0].to_dict()
                                                      if (chosen_role and not vac_df.empty) else {})
            feat = build_feature_row(chosen_role, corpus_text, vac_row, sector=current_sector)
            base_prob = predict_prob(feat, sector=current_sector)
            _w = get_sector_weights(current_sector)
            _blend = st.session_state.get('blend', 0.4)
            adj_prob = adjust_with_custom_factors(base_prob, feat, _w, _blend)
            client = get_openai_client(openai_key)
            narrative = gpt_narrative_and_qa(client, lang, chosen_role, adj_prob, feat, vac_row,
                                             question=None, corpus_text=corpus_text, retrieved=None)
            with st.chat_message("assistant"):
                st.markdown(narrative)
            current_meta.setdefault("messages", []).append(("assistant", narrative))
            # ---- SYNC ----
            candidate_id = current_meta.get("display_name") or (cv_names[0] if cv_names else "Candidate")
            narrative_clean = re.sub(r"(\n\s*){2,}", "\n\n", narrative.strip())
            st.session_state.setdefault("ai_narratives", {})[candidate_id] = narrative_clean
            current_meta["narrative"] = narrative_clean
            st.success(f"✅ Narrative stored and synced for {candidate_id}.")

        # ----- BATCH NARRATIVES -----
        st.markdown("---")
        st.subheader("Batch: Instant Narratives for Uploaded CVs")
        st.caption("Generates per-CV narratives using the matched vacancy.")
        if st.button("Generate Narratives for All Uploaded CVs", key="batch_cv_narratives"):
            if not cv_files:
                st.warning("Please upload one or more CV files first.")
            else:
                results = []
                vacancy_txt = st.session_state.get("auto_vac_text", "")
                chosen_role = current_meta.get("role") or role
                vac_row = current_meta.get("vac_row") or (vac_df[vac_df["JobTitle"] == chosen_role].iloc[0].to_dict()
                                                          if (chosen_role and not vac_df.empty) else {})
                cover_text_for_all = basic_clean(cl_text_area) if cl_text_area else ""
                for up in cv_files:
                    try:
                        res = generate_narrative_for_single_cv(upload_file=up, role=chosen_role,
                            vac_row=vac_row, vacancy_txt=vacancy_txt, lang=lang,
                            sector_for_model=current_sector, openai_key=openai_key,
                            cover_letter_text=cover_text_for_all)
                        results.append(res)
                        with st.expander(f"📄 {res['filename']} · Success {res['pred_success_adj']*100:.1f}%"):
                            st.markdown(res["narrative"])
                    except Exception as e:
                        st.error(f"Error processing {getattr(up, 'name', 'file')}: {e}")

                if results:
                    df_batch = pd.DataFrame(results)
                    st.download_button("Download batch narratives as CSV",
                                       data=df_batch.to_csv(index=False).encode("utf-8"),
                                       file_name="batch_narratives.csv",
                                       mime="text/csv")
                    st.success(f"Generated {len(results)} narratives.")

        # ----- WHAT-IF + SHAP EXPLAINABILITY -----
        st.markdown("---")
        cols = st.columns([1, 1])
        with cols[0]:
            st.subheader(t("what_if", lang))
            sal = st.slider(t("salary_boost", lang), 0, 30, 5, 1, key="sal_chat")
            rem = st.slider(t("remote_days", lang), 0, 5, 2, 1, key="rem_chat")
            if 'last_features' in st.session_state:
                p0 = st.session_state.get('last_probability', 0.0)
                p_offer = offer_uplift(p0, sal, rem)
                _w = get_sector_weights(current_sector)
                _blend = st.session_state.get('blend', 0.4)
                adj = adjust_with_custom_factors(p_offer, st.session_state['last_features'], _w, _blend)
                st.metric(t("offer_uplift", lang), f"{adj*100:.1f}%")
                acc0 = acceptance_probability(p0, st.session_state['last_features'], 0, 0)
                acc1 = acceptance_probability(adj, st.session_state['last_features'], sal, rem)
                st.metric("Acceptance (est.)", f"{acc1*100:.1f}%", delta=f"{(acc1-acc0)*100:.1f} pp")
            else:
                st.caption("Upload a CV and select a vacancy to simulate.")

        with cols[1]:
            st.subheader(t("shap_title", lang))
            try:
                if 'last_features' in st.session_state:
                    row = pd.DataFrame([st.session_state['last_features']])
                    for c in feature_cols:
                        if c.startswith("Sector_"):
                            row[c] = 0
                    sec_col = f"Sector_{current_sector}"
                    if sec_col in feature_cols:
                        row[sec_col] = 1
                    for c in feature_cols:
                        if c not in row.columns:
                            row[c] = 0
                    row = row[feature_cols]
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(row)
                    fig = plt.figure()
                    shap.waterfall_plot(
                        shap.Explanation(values=sv[0], base_values=explainer.expected_value,
                                         data=row.iloc[0].values, feature_names=row.columns.tolist()),
                        show=False
                    )
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                else:
                    st.caption("Upload a CV to view per-candidate SHAP.")
            except Exception:
                st.caption("SHAP explanation unavailable in this environment.")

# ===== EXPLORER =====
with section[2]:
    st.title(t("title_explorer", lang))
    st.markdown("<div class='personato-card'>", unsafe_allow_html=True)
    fig2 = px.scatter(
        df, x="ExperienceYears", y="SkillMatch", size="CultureFit", color="Hired",
        title="Experience vs Skill Match (bubble = CultureFit)"
    )
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=TEXT_LIGHT,
        title=dict(font=dict(color=TEXT_LIGHT)), legend=dict(font=dict(color=TEXT_LIGHT)),
        xaxis=dict(title_font=dict(color=TEXT_LIGHT), tickfont=dict(color=TEXT_LIGHT)),
        yaxis=dict(title_font=dict(color=TEXT_LIGHT), tickfont=dict(color=TEXT_LIGHT)),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.dataframe(df.sample(25), use_container_width=True)
    if 'last_features' in st.session_state:
        st.markdown(f"#### Candidate Insights: **{st.session_state.get('last_candidate_name','Candidate')}**  ·  Role: **{st.session_state.get('last_role','-')}**")
        _pd = pd.DataFrame([st.session_state['last_features']])
        st.dataframe(_pd, use_container_width=True)
        
# ===== ASSESSMENT & INSIGHTS =====
with section[3]:
    import base64, os, json, re, datetime, plotly.io as pio, pdfkit
    import plotly.graph_objects as go
    import numpy as np
    from openai import OpenAI

    # ---------- THEME ----------
    PRIMARY_BG, ACCENT, TEXT = "#0B1A33", "#FFD700", "#FFFFFF"
    st.markdown(
        f"<style>body{{background:{PRIMARY_BG};color:{TEXT};}}</style>",
        unsafe_allow_html=True,
    )

    # ---------- HEADER ----------
    live_sync = st.session_state.get("live_sync", False)
    header_html = (
        f"<h1 style='color:{ACCENT};'>🧠 Assessment & Insights "
        + (
            f"<span style='font-size:0.8em;color:{ACCENT};'>⚡ Live Sync Active</span>"
            if live_sync
            else ""
        )
        + "</h1>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # ---------- LOCALIZATION ----------
    lang = st.session_state.get("lang", "en")
    L = {
        "intro": {
            "en": "Run independent assessments for multiple candidates. Results are stored, visualized, and exportable per candidate.",
            "nl": "Voer onafhankelijke beoordelingen uit voor meerdere kandidaten. Resultaten worden per kandidaat opgeslagen, weergegeven en geëxporteerd.",
        },
        "form_title": {
            "en": "Candidate Assessment Form",
            "nl": "Kandidaatbeoordelingsformulier",
        },
        "submit": {"en": "Submit Assessment", "nl": "Verstuur Beoordeling"},
        "manage": {"en": "Manage All Candidates", "nl": "Beheer Alle Kandidaten"},
        "completed": {
            "en": "✅ Assessment completed — AI computed Assessment Fit",
            "nl": "✅ Beoordeling voltooid — AI berekende Beoordelingsscore",
        },
        "export": {"en": "📄 Export PDF", "nl": "📄 Exporteer PDF"},
        "fit_label": {"en": "Assessment Fit", "nl": "Beoordelingsscore"},
        "archived": {"en": "Assessment archived.", "nl": "Beoordeling gearchiveerd."},
        "deleted": {"en": "Assessment deleted.", "nl": "Beoordeling verwijderd."},
    }
    st.markdown(L["intro"][lang])

    # ---------- INIT STORAGE ----------
    st.session_state.setdefault("candidate_assessments", {})
    st.session_state.setdefault("archived_assessments", {})
    st.session_state.setdefault("ai_narratives", {})  # synced with Conversational Recruiter

    # ---------- CANDIDATE SELECTION ----------
    st.markdown("### 👤 Candidate Selection")
    all_names = list(st.session_state["candidate_assessments"].keys())
    mode = st.radio(
        "Choose mode:", ["Select existing candidate", "Add new candidate"], horizontal=True
    )
    if mode == "Select existing candidate" and all_names:
        candidate_name = st.selectbox("Select candidate", all_names)
    else:
        candidate_name = st.text_input("Enter candidate name", value="")

    if not candidate_name.strip():
        st.info("Enter or select a candidate to begin assessment.")
        candidate_name = None
    else:
        # ---------- CONTEXT ----------
        vacancy_txt = st.session_state.get("auto_vac_text", "")
        role = st.session_state.get("vacancy_select", "Data Analyst")
        openai_key = st.session_state.get("openai_key", os.getenv("OPENAI_API_KEY", ""))

        # ---------- AI QUESTION GENERATOR ----------
        def generate_ai_questions(vac_text, role_name, lang="en", api_key=None):
            """Semantic recruiter-grade AI question generator."""
            try:
                if not api_key:
                    raise ValueError("No OpenAI key.")
                client = OpenAI(api_key=api_key)
                prompt = f"""
                You are a professional recruiter. Generate 6–8 open-ended, high-quality assessment questions covering:
                1. Motivation / ambition
                2. Technical / role-fit skills
                3. Experience relevance
                4. Behavioral & social dynamics
                5. Cultural alignment
                6. Growth / potential
                7. A reflective 'what-if' scenario
                Role: {role_name}
                Language: {"Dutch" if lang == "nl" else "English"}
                VACANCY:
                {vac_text[:2500]}
                """
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert HR recruiter creating contextual interview questions.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.6,
                    max_tokens=600,
                )
                txt = res.choices[0].message.content
                questions = [
                    q.strip("1234567890. ").strip()
                    for q in txt.split("\n")
                    if len(q.strip()) > 10
                ]
                return questions[:8]
            except Exception:
                return [
                    f"What motivated you to apply for the {role_name} position?",
                    "Which skills make you a strong fit for this role?",
                    "Describe a past experience that shows your competence.",
                    "How do you collaborate with others at work?",
                    "What about our company culture appeals to you?",
                    "Where do you see your growth potential?",
                    "Imagine getting this job — what would you change or optimize first?",
                ]

        if st.button("🔄 Generate AI Questions"):
            st.session_state["assessment_questions"] = generate_ai_questions(
                vacancy_txt, role, lang, openai_key
            )
            st.success("✅ AI recruiter-grade questions generated.")

        qs = st.session_state.get(
            "assessment_questions",
            generate_ai_questions(vacancy_txt, role, lang, openai_key),
        )

        # ---------- ASSESSMENT FORM ----------
        st.markdown(f"### {L['form_title'][lang]} — *{candidate_name}*")
        responses = {}
        with st.form(f"form_{candidate_name}"):
            for q in qs:
                responses[q] = st.text_area(
                    q, height=100, key=f"resp_{candidate_name}_{hash(q)}"
                )
            submit = st.form_submit_button(L["submit"][lang])

        # ---------- RADAR ----------
        def build_radar(scores_dict, title="Assessment"):
            cats = ["Motivation", "Joy", "Trust", "AssessmentFit"]
            vals = [scores_dict.get(c, 0) for c in cats] + [
                scores_dict.get("Motivation", 0)
            ]
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=cats + [cats[0]],
                    fill="toself",
                    name=title,
                    line_color=ACCENT,
                    fillcolor="rgba(255,215,0,0.3)",
                )
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=ACCENT)
                ),
                showlegend=False,
                paper_bgcolor=PRIMARY_BG,
                font_color=TEXT,
            )
            return fig

        # ---------- EXPORT REPORT ----------
        def export_report(assessment, candidate_name, role="Candidate"):
            import base64, os, datetime, plotly.io as pio

            timestamp = assessment["timestamp"].replace(":", "-").replace(" ", "_")
            file_prefix = f"{candidate_name}_{timestamp}"

            # ✅ Fetch AI narrative
            narratives_dict = st.session_state.get("ai_narratives", {})
            narrative_text = ""
            for k, v in narratives_dict.items():
                if candidate_name.lower() in k.lower():
                    narrative_text = v
                    break
            if not narrative_text:
                narrative_text = (
                    "⚠️ No AI narrative found. Please run the Conversational Recruiter for this candidate."
                )

            # ✅ Precompute safe HTML version to avoid backslash in f-string
            narrative_html = narrative_text.replace("\n", "<br>")

            # ---------- RADAR IMAGE ----------
            try:
                fig = build_radar(assessment["scores"])
                img_b64 = base64.b64encode(pio.to_image(fig, format="png")).decode()
            except Exception:
                img_b64 = ""

            # ---------- LOGO ----------
            logo_path = r"C:\Users\DavidAdewunmi\Downloads\CynthAI_Logo.png"
            logo_b64 = ""
            if os.path.exists(logo_path):
                with open(logo_path, "rb") as f:
                    logo_b64 = base64.b64encode(f.read()).decode()

            # ---------- HTML REPORT ----------
            html = f"""
            <html>
            <body style='background:{PRIMARY_BG};color:{TEXT};font-family:Arial;padding:25px;'>
              {'<img src="data:image/png;base64,' + logo_b64 + '" width="160">' if logo_b64 else ''}
              <h2 style='color:{ACCENT};text-align:center;'>Assessment Report — {candidate_name} ({role})</h2>
              <p><b>Timestamp:</b> {timestamp}</p>

              <h3 style='color:{ACCENT};'>AI Narrative Summary</h3>
              <div style='background:rgba(255,255,255,0.05);padding:12px;border-left:4px solid {ACCENT};
                          border-radius:8px;margin-bottom:20px;'>
                  {narrative_html}
              </div>

              <h3 style='color:{ACCENT};'>Assessment Scores</h3>
              <ul>
                <li>Assessment Fit: {assessment['scores'].get('AssessmentFit', 0)*100:.1f}%</li>
                <li>Motivation: {assessment['scores'].get('Motivation', 0)*100:.1f}%</li>
                <li>Joy: {assessment['scores'].get('Joy', 0)*100:.1f}%</li>
                <li>Trust: {assessment['scores'].get('Trust', 0)*100:.1f}%</li>
              </ul>
              {'<img src="data:image/png;base64,' + img_b64 + '" width="450">' if img_b64 else ''}

              <h3 style='color:{ACCENT};margin-top:25px;'>Candidate Responses</h3>
              <ul>{''.join([f'<li><b>{q}</b>: {a}</li>' for q, a in assessment['answers'].items()])}</ul>

              <p style='text-align:center;color:{ACCENT};margin-top:25px;'>
                Generated by <b>CynthAI© TalentLens</b> — Powered by Crowe Technology
              </p>
            </body></html>
            """

            # ---------- PDF EXPORT (Streamlit Cloud Safe) ----------
         # ---------- PDF EXPORT ----------
            try:
                from fpdf import FPDF
                import io

                pdf = FPDF()
                pdf.add_page()

                # --- Add CynthAI Banner ---
                pdf.set_fill_color(0, 0, 0)  # Black header background
                pdf.rect(0, 0, 210, 20, "F")
                pdf.set_text_color(255, 215, 0)  # Gold title text
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 12, "CynthAI© TalentLens — AI Narrative Report", ln=1, align="C")
                pdf.ln(10)

                # --- Body (Narrative Text) ---
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", "", 12)
                narrative_text = html  # reuse your AI narrative or text variable
                pdf.multi_cell(
                    0, 8,
                    narrative_text.replace("<br>", "\n").replace("<p>", "").replace("</p>", "")
                )

                # --- Footer ---
                pdf.ln(10)
                pdf.set_text_color(100, 100, 100)
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(
                    0, 10,
                    "Powered by CynthAI© TalentLens — Crowe Technology",
                    ln=1, align="C"
                )

                # --- Export to Bytes and Streamlit Download ---
                buf = io.BytesIO()
                pdf.output(buf)
                buf.seek(0)

                st.download_button(
                    label=f"📄 Download {file_prefix}.pdf",
                    data=buf,
                    file_name=f"{file_prefix}.pdf",
                    mime="application/pdf",
                )
                st.success("✅ PDF successfully generated with AI Narrative.")

            except Exception as e:
                st.warning(f"⚠️ PDF export failed ({e}); fallback to HTML.")
                html_b64 = base64.b64encode(html.encode()).decode()
                st.markdown(
                    f'<a href="data:text/html;base64,{html_b64}" download="{file_prefix}.html" '
                    f'style="color:{ACCENT};font-weight:600;">📝 Download HTML</a>',
                    unsafe_allow_html=True,
                )
        # ---------- SUBMIT ----------
        if submit:
            valid = {q: a.strip() for q, a in responses.items() if a.strip()}
            if not valid:
                st.warning("⚠️ Please answer at least one question.")
            else:
                combined = " ".join(valid.values())
                mot, emo = sentiment_score(combined), emotion_vector(combined)
                assess_fit = float(
                    np.clip(
                        0.6 * mot + 0.4 * ((emo.get("joy", 0) + emo.get("trust", 0)) / 2),
                        0,
                        1,
                    )
                )
                new = {
                    "timestamp": str(datetime.datetime.now())[:19],
                    "answers": valid,
                    "scores": {
                        "Motivation": mot,
                        "Joy": emo.get("joy", 0),
                        "Trust": emo.get("trust", 0),
                        "AssessmentFit": assess_fit,
                    },
                }
                st.session_state["candidate_assessments"].setdefault(
                    candidate_name, []
                ).append(new)
                st.session_state["last_candidate"] = candidate_name
                st.session_state["last_assessment"] = new
                st.success(f"{L['completed'][lang]} — {assess_fit*100:.1f}%")

        # ---------- VISUALIZE ----------
        if st.session_state.get("last_candidate") == candidate_name:
            last = st.session_state.get("last_assessment", {})
            if last:
                st.plotly_chart(build_radar(last["scores"]), use_container_width=True)
                st.markdown("### Export Candidate Report")
                export_report(last, candidate_name, role=role)

    # ---------- MANAGE ALL CANDIDATES ----------
    st.markdown(f"### {L['manage'][lang]}")
    if st.session_state["candidate_assessments"]:
        for cname, records in st.session_state["candidate_assessments"].items():
            with st.expander(f"🧩 {cname} — {len(records)} assessment(s)", expanded=False):
                for idx, rec in enumerate(records[::-1]):
                    ts = rec["timestamp"]
                    fit = rec["scores"]["AssessmentFit"]
                    st.markdown(f"**{ts}** — Fit: {fit*100:.1f}%")
                    st.plotly_chart(build_radar(rec["scores"]), use_container_width=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("📦 Archive", key=f"arch_{cname}_{idx}"):
                            st.session_state["archived_assessments"].setdefault(
                                cname, []
                            ).append(rec)
                            st.session_state["candidate_assessments"][cname].remove(rec)
                            st.success(L["archived"][lang])
                    with c2:
                        if st.button("🗑 Delete", key=f"del_{cname}_{idx}"):
                            st.session_state["candidate_assessments"][cname].remove(rec)
                            st.warning(L["deleted"][lang])
                    with c3:
                        if st.button("📄 Export", key=f"exp_{cname}_{idx}"):
                            export_report(rec, cname, role="Candidate")
    else:
        st.caption("No assessments stored yet — submit one to begin.")
    
   # ===== CUSTOM FACTORS =====
with section[4]:
    st.title("⚙️ Custom Factors")

    st.markdown("""
    Define weighting parameters that directly influence the **Assessment & Insights** engine.  
    These weights affect how *Motivation*, *Fit*, and *Narrative reasoning* are calculated for each candidate.
    """)

    # --- Initialize weights if not found ---
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "IT": {"Experience": 0.25, "SkillMatch": 0.25, "Motivation": 0.2, "CultureFit": 0.15, "Sentiment": 0.1, "Emotion": 0.05},
            "Finance": {"Experience": 0.3, "SkillMatch": 0.25, "Motivation": 0.2, "CultureFit": 0.1, "Sentiment": 0.1, "Emotion": 0.05},
            "HR": {"Experience": 0.2, "SkillMatch": 0.25, "Motivation": 0.25, "CultureFit": 0.15, "Sentiment": 0.1, "Emotion": 0.05}
        }

    # --- Select sector ---
    sectors = list(st.session_state.weights.keys())
    selected_sector = st.selectbox("Select sector", sectors, index=0)

    st.subheader(f"🎯 Weights for {selected_sector} sector")

    factors = st.session_state.weights[selected_sector]
    new_factors = {}

    st.caption("Adjust factor influence on prediction and AI-driven assessment:")

    cols = st.columns(3)
    with cols[0]:
        new_factors["Experience"] = st.slider("Experience", 0.0, 1.0, float(factors["Experience"]), 0.05)
        new_factors["SkillMatch"] = st.slider("Skill Match", 0.0, 1.0, float(factors["SkillMatch"]), 0.05)
    with cols[1]:
        new_factors["Motivation"] = st.slider("Motivation", 0.0, 1.0, float(factors["Motivation"]), 0.05)
        new_factors["CultureFit"] = st.slider("Culture Fit", 0.0, 1.0, float(factors["CultureFit"]), 0.05)
    with cols[2]:
        new_factors["Sentiment"] = st.slider("Sentiment", 0.0, 1.0, float(factors["Sentiment"]), 0.05)
        new_factors["Emotion"] = st.slider("Emotion", 0.0, 1.0, float(factors["Emotion"]), 0.05)

    normalize = st.checkbox("Normalize weights (sum = 1)", value=True)
    if normalize:
        total = sum(new_factors.values())
        if total > 0:
            new_factors = {k: v / total for k, v in new_factors.items()}

    # Save back to session
    st.session_state.weights[selected_sector] = new_factors

    # --- Blend Control ---
    st.markdown("---")
    st.subheader("🔀 Blend Factor")
    st.session_state["blend_factor"] = st.slider(
        "Blend with model prediction",
        0.0, 1.0,
        float(st.session_state.get("blend_factor", 0.5)), 0.05,
        help="0 = model only, 1 = fully custom weighting"
    )

    # --- Live Sync toggle ---
    st.markdown("---")
    live_sync = st.toggle("🔄 Live Sync with Assessment & Insights", value=st.session_state.get("live_sync", True))
    st.session_state["live_sync"] = live_sync

    if live_sync:
        st.info("✅ Live Sync is ON — changes instantly update the Assessment & Insights tab.")
        # Mark for assessment recalculation
        st.session_state["trigger_assessment_refresh"] = True
    else:
        st.warning("⏸️ Live Sync is OFF — weights will apply after manual refresh.")

    st.success("✅ Custom factors and blend are stored for this session.")

    # --- Display overview ---
    st.markdown("### 📊 Overview of All Sector Weights")
    df_w = pd.DataFrame(st.session_state.weights).T
    st.dataframe(df_w.style.format("{:.2f}"), use_container_width=True)
     
    
# ===== BIAS & EXPLAINABILITY =====
# ===== BIAS & EXPLAINABILITY =====
with section[5]:
    st.title(t("title_bias", lang))
    st.caption("This section provides per-candidate model explainability and bias insights based on the last analyzed narrative.")

    # --- Retrieve current candidate context ---
    last_name = st.session_state.get("last_candidate_name", None)
    feat = st.session_state.get("last_features", None)
    prob = st.session_state.get("last_probability", None)
    role = st.session_state.get("last_role", "-")
    sector = st.session_state.get("last_sector", get_current_sector())
    narrative_text = st.session_state.get("ai_narratives", {}).get(last_name, "")

    # --- Conditional Render ---
    if not feat or not last_name:
        st.warning("No candidate analysis found yet. Please generate a narrative first in the Conversational Recruiter tab.")
    else:
        # --- Display Narrative Summary ---
        st.subheader(f"🧠 Narrative Summary — {last_name}")
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);padding:1rem;border-left:4px solid {ACCENT_RED};border-radius:8px;'>"
            f"{narrative_text or 'No stored narrative. Run Conversational Recruiter to sync results.'}</div>",
            unsafe_allow_html=True
        )

        # --- SHAP Explanation for this Candidate ---
        st.markdown("### 🔍 Explainability — SHAP Feature Contribution")
        try:
            row = pd.DataFrame([feat])
            for c in feature_cols:
                if c.startswith("Sector_"):
                    row[c] = 0
            sec_col = f"Sector_{sector}"
            if sec_col in feature_cols:
                row[sec_col] = 1
            for c in feature_cols:
                if c not in row.columns:
                    row[c] = 0
            row = row[feature_cols]

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(row)
            base_val = explainer.expected_value
            fig = plt.figure()
            shap.waterfall_plot(
                shap.Explanation(values=sv[0], base_values=base_val,
                                 data=row.iloc[0].values, feature_names=row.columns.tolist()),
                show=False
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Unable to compute SHAP explanation: {e}")

        # --- Top Positive / Negative Contributions ---
        try:
            shap_vals = pd.Series(sv[0], index=row.columns)
            shap_sorted = shap_vals.abs().sort_values(ascending=False).head(8)
            st.markdown("### 📊 Top Feature Influences")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top Positive Features**")
                st.dataframe(shap_vals.sort_values(ascending=False).head(5).to_frame("Impact"), use_container_width=True)
            with c2:
                st.markdown("**Top Negative Features**")
                st.dataframe(shap_vals.sort_values(ascending=True).head(5).to_frame("Impact"), use_container_width=True)
        except Exception:
            pass

        # --- Fairness Check ---
        st.markdown("### ⚖️ Fairness / Bias Context")
        df_sector = df[df["Sector"] == sector]
        mean_success = df_sector["Hired"].mean()
        feat_names = ["MotivationScore","SkillMatch","CultureFit","SentimentScore","EmotionPos"]
        ref_means = {f: df_sector[f].mean() for f in feat_names if f in df_sector}
        st.markdown(
            f"**Sector baseline success rate:** {mean_success*100:.1f}%  |  "
            f"**Candidate predicted success:** {prob*100:.1f}%"
        )

        bias_df = pd.DataFrame([
            {"Feature": f, "Candidate": feat.get(f, 0.0), "Sector Avg": ref_means.get(f, 0.0)}
            for f in feat_names if f in feat
        ])
        st.bar_chart(bias_df.set_index("Feature"))

        # --- AI Explainability Narrative ---
        st.markdown("### 🧩 AI Bias & Explainability Summary")
        client = get_openai_client(openai_key)
        if client:
            explain_prompt = f"""
            You are an AI ethics auditor. Explain, in plain language, how the candidate's narrative and model features
            might indicate potential bias or fairness issues. Compare against the sector baseline averages.
            Candidate role: {role}, Sector: {sector}.
            Candidate narrative:
            {narrative_text[:1500]}
            Feature summary: {json.dumps(feat, indent=2)}
            """
            try:
                exp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in AI fairness and HR explainability."},
                        {"role": "user", "content": explain_prompt}
                    ],
                    temperature=0.5, max_tokens=500
                )
                st.markdown(exp.choices[0].message.content)
            except Exception:
                st.caption("AI-based bias explanation unavailable (check API key).")
        else:
            st.caption("No API key found — upload key in the sidebar to generate explainability summary.")

# CLOSE Bias & Explainability block completely
# -------------------------------------------------------------


# ===== SETTINGS & DATA =====
with section[6]:
    st.title(t("title_settings", lang))
    st.write("Vacancies in use:")
    st.dataframe(vac_df, use_container_width=True)

    st.write("Model training set (sample):")
    st.dataframe(df.sample(20), use_container_width=True)

    st.caption("ℹ️ For weight tuning, go to the ⚙️ Custom Factors tab.")

    st.markdown("---")
    st.subheader(t("msg_outreach", lang))

    cname = st.text_input(t("candidate_name", lang), st.session_state.get("last_candidate_name", "Jane Doe"), key="out_name")
    rtitle = st.text_input(t("role_title", lang), st.session_state.get("last_role", "Data Analyst"), key="out_role")
    fit_demo = st.session_state.get("last_features", {}).get("CultureFit", 0.7)
    prob_demo = st.session_state.get("last_probability", 0.8)

    def gen_outreach(lang, name, role, prob, fit):
        if lang == "nl":
            return (
                f"Hi {name},\n\nJouw profiel sluit goed aan op onze rol **{role}**. "
                f"Wat opvalt: sterke motivatie en een cultuurfit van {fit:.0%}. "
                f"Zullen we een (online) kennismaking plannen? Welke dag past voor jou?\n\nGroet,\nTeam Personato"
            )
        return (
            f"Hi {name},\n\nYour profile aligns well with our **{role}**. "
            f"What stands out: strong motivation and a culture fit of {fit:.0%}. "
            f"Shall we schedule a quick intro call? What day works for you?\n\nBest,\nPersonato Team"
        )

    if st.button(t("generate", lang), key="out_btn"):
        st.text_area("Message", gen_outreach(lang, cname, rtitle, prob_demo, fit_demo), height=180)
        st.caption("✍️ " + t("copy_hint", lang))
        
# ===== 🌐 LIVE VACANCIES PORTAL (FINAL — FULL AI INTEGRATION + TOOLTIP HOVER GLOW) =====
import io, base64, requests, xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import streamlit as st

# ---------- 💎 CSS for Hover Glow + Tooltip ----------
st.markdown("""
<style>
.platform-logo {
    display: inline-block;
    text-align: center;
    margin: 0 auto;
    position: relative;
}
.platform-logo img {
    transition: transform 0.3s ease-in-out, filter 0.3s ease-in-out;
    cursor: pointer;
}
.platform-logo img:hover {
    filter: drop-shadow(0 0 8px gold);
    transform: scale(1.05);
}
.platform-logo .tooltip {
    visibility: hidden;
    background-color: #111;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 6px;
    position: absolute;
    z-index: 1;
    bottom: 110%;
    left: 50%;
    margin-left: -60px;
    width: 120px;
    font-size: 0.75rem;
    opacity: 0;
    transition: opacity 0.3s;
}
.platform-logo:hover .tooltip {
    visibility: visible;
    opacity: 0.9;
}
.caption {
    text-align: center;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# -# ===== 🌐 LIVE VACANCIES PORTAL (FINAL SHOWCASE VERSION) =====
import io, base64, requests, xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import streamlit as st

# ---------- 💎 Enhanced CSS (Hover Info Cards + Center Layout) ----------
st.markdown("""
<style>
.platform-logo {
    display: inline-block;
    text-align: center;
    margin: 10px 20px;
    position: relative;
}
.platform-logo img {
    transition: all 0.35s ease-in-out;
    cursor: pointer;
    filter: brightness(0.95) saturate(1.1);
}
.platform-logo img:hover {
    transform: scale(1.08) translateY(-4px);
    filter: brightness(1.2) drop-shadow(0 0 10px gold) drop-shadow(0 0 20px #FFD700);
}
.platform-logo .info-card {
    visibility: hidden;
    background-color: rgba(26,26,26,0.95);
    color: gold;
    text-align: left;
    border-radius: 8px;
    padding: 8px 10px;
    position: absolute;
    z-index: 2;
    width: 180px;
    bottom: 120%;
    left: 50%;
    margin-left: -90px;
    font-size: 0.75rem;
    box-shadow: 0 0 6px rgba(255,215,0,0.3);
    opacity: 0;
    transition: opacity 0.3s ease-in-out, transform 0.3s ease-in-out;
    transform: translateY(6px);
}
.platform-logo:hover .info-card {
    visibility: visible;
    opacity: 1;
    transform: translateY(0);
}
.caption {
    text-align: center;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 4px;
}
.source-box {
    width: 70%;
    margin-left: auto;
    margin-right: auto;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- 🖼️ Universal Logo Renderer (Foolproof Base64 + Glow + Tooltip) ----------
def logo_display(url, width=100, caption="", info=None):
    """
    Reliable logo display: handles SVG/PNG, auto-recolors dark SVGs for dark mode,
    adds gold hover glow and info card.
    """
    try:
        r = requests.get(url, timeout=10, stream=True)
        if r.status_code == 200:
            content_type = r.headers.get("content-type", "")
            data = r.content

            # Convert to base64
            b64_data = base64.b64encode(data).decode("utf-8")

            if "svg" in content_type or url.lower().endswith(".svg"):
                mime = "image/svg+xml"
            else:
                mime = "image/png"

            info_html = "<br>".join(info) if info else ""
            html = f"""
            <div class="platform-logo">
                <div class="info-card">{info_html}</div>
                <img src="data:{mime};base64,{b64_data}" width="{width}">
                <div class="caption">{caption}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        else:
            raise Exception(f"Bad status {r.status_code}")
    except Exception as e:
        st.markdown(
            f"<div class='platform-logo' style='color:gold;text-align:center;'>⚠️ {caption}</div>",
            unsafe_allow_html=True,
        )

# ---------- 🌍 Multi-Source Vacancy Connector ----------
def fetch_live_vacancies(source, company, api_key=None):
    try:
        source = source.lower()
        if source == "recruitee":
            r = requests.get(f"https://{company}.recruitee.com/api/offers/")
            return pd.DataFrame(r.json().get("offers", []))
        elif source == "lever":
            r = requests.get(f"https://api.lever.co/v0/postings/{company}")
            return pd.DataFrame(r.json())
        elif source == "greenhouse":
            r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs")
            return pd.DataFrame(r.json().get("jobs", []))
        elif source == "personio":
            r = requests.get(f"https://{company}.jobs.personio.de/xml")
            root = ET.fromstring(r.text)
            data = []
            for job in root.findall(".//job"):
                data.append({
                    "title": job.findtext("name"),
                    "location": job.findtext("office"),
                    "url": job.findtext("url"),
                    "description": job.findtext("description", "")
                })
            return pd.DataFrame(data)
        elif source == "adzuna":
            if not api_key or ":" not in api_key:
                st.warning("Adzuna requires app_id:app_key format.")
                return pd.DataFrame()
            app_id, app_key = api_key.split(":")
            r = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/nl/search/1?"
                f"app_id={app_id}&app_key={app_key}&results_per_page=25"
            )
            jobs = r.json().get("results", [])
            return pd.DataFrame([{
                "title": j["title"],
                "location": j["location"]["display_name"],
                "url": j["redirect_url"],
                "description": j.get("description", "")
            } for j in jobs])
        elif source in ["linkedin", "indeed"]:
            st.warning(f"{source.title()} API is restricted — manual or partner integration needed.")
        elif source == "monsterboard":
            return pd.DataFrame([{
                "title": "Audit Consultant (demo)",
                "location": "Eindhoven",
                "url": "https://www.monsterboard.nl/vacatures/",
                "description": "Demo placeholder — replace with licensed API."
            }])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching from {source}: {e}")
        return pd.DataFrame()

# ---------- 🧠 AI Narrative Wrapper + Predictive Integration ----------
def generate_narrative(text, fit_score):
    try:
        client = get_openai_client(openai_key)
        lang = st.session_state.get("lang", "en")
        role = st.session_state.get("vacancy_select", "Candidate Role")
        vac_row = st.session_state.get("vac_row", {})
        feat = build_feature_row(role, text, vac_row, sector=st.session_state.get("sector", "General"))
        return gpt_narrative_and_qa(
            client, lang, role, fit_score, feat, vac_row,
            question=None, corpus_text=text, retrieved=None
        )
    except Exception as e:
        st.error(f"❌ Narrative generation failed: {e}")
        return "Narrative unavailable."

def run_full_analysis(vacancy_text):
    try:
        st.info("⚙️ Running AI analysis on attached vacancy ...")
        cv_text = st.session_state.get("cv_text", "")
        combined = f"{vacancy_text}\n\n{cv_text}"
        base_prob = predict_prob(combined)
        sector = st.session_state.get("sector", "General")
        vac_row = st.session_state.get("vac_row", {})
        role = st.session_state.get("vacancy_select", "Candidate Role")
        feat = build_feature_row(role, combined, vac_row, sector)
        w = get_sector_weights(sector)
        blend = st.session_state.get("blend", 0.4)
        adj = adjust_with_custom_factors(base_prob, feat, w, blend)
        acc = acceptance_probability(adj, feat)
        narrative = generate_narrative(combined, adj)
        cand = st.session_state.get("last_candidate_name", "Candidate")
        st.session_state.update({"fit_score": adj, "narrative_text": narrative, "auto_vac_text": vacancy_text})
        st.session_state.setdefault("ai_narratives", {})[cand] = narrative

        st.success("✅ AI analysis complete — summary below")
        with st.expander("🧠 AI Narrative & Fit Summary", expanded=True):
            st.metric("Candidate Fit Score", f"{adj*100:.1f}%")
            st.metric("Acceptance (Est.)", f"{acc*100:.1f}%")
            st.markdown("**Narrative:**")
            st.markdown(narrative)
            st.caption("🔍 Generated by CynthAI© TalentLens predictive engine.")
    except Exception as e:
        st.error(f"❌ AI analysis failed: {e}")

# ---------- 🎯 STREAMLIT TAB ----------
with section[7]:
    st.title("🌐 Live Vacancies Portal")
    st.caption("Connect to global job boards and ATS platforms — fetch, attach & auto-analyze using TalentLens A.I.")

    # --- Platform Logos with Info Cards ---
    st.markdown("### Integrated Platforms")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        logo_display("https://cdn.creazilla.com/icons/3204991/logo-linkedin-icon-sm.png", 100, "LinkedIn",
                     ["🌍 Global network", "🔑 Restricted API", "🧠 Integration: Manual Partner"])
    with c2:
        logo_display("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzM4VuvVEOGDPxMMxlPJl0zTLZzhSMvi5jTQ&s", 100, "Indeed",
                     ["🌍 Global reach", "🔑 Public search API", "🧠 Direct data integration"])
    with c3:
        logo_display("https://www.team4recruiters.nl/wp-content/uploads/2019/02/Monsterboard.png", 100,
                     "Monsterboard", ["🌍 Global platform", "🔑 Open RSS Feed", "🧠 Data feed integration"])
    with c4:
        logo_display("https://upload.wikimedia.org/wikipedia/commons/a/ad/Recruitee_logo.png", 100, "Recruitee",
                     ["🇳🇱 Europe / Global", "🔑 Public REST API", "🧠 Full TalentLens sync"])

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        logo_display("https://upload.wikimedia.org/wikipedia/commons/f/fd/Greenhouse_logo.png", 100, "Greenhouse",
                     ["🌍 US/EU", "🔑 JSON API", "🧠 Predictive mapping enabled"])
    with c6:
        logo_display("https://www.lever.co/wp-content/uploads/2024/05/LinkedIn-Apply-Connect-2-1.png", 100, "Lever",
                     ["🌍 Global", "🔑 REST API", "🧠 TalentLens AI link"])
    with c7:
        logo_display("https://www.datocms-assets.com/85623/1719638527-intigriti_blog_customer-stories_spotlight_personio_people_power-1.png?auto=format", 90,
                     "Personio", ["🇩🇪 EU focus", "🔑 XML feed", "🧠 ATS analytics connected"])
    with c8:
        logo_display("https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/adzuna.svg", 100,
                     "Adzuna", ["🇬🇧 Global", "🔑 app_id + app_key", "🧠 Job market API"])

    st.divider()

    # --- Centered Input Section ---
    st.markdown('<div class="source-box">', unsafe_allow_html=True)
    src = st.selectbox("Select Source", [
        "Recruitee", "Lever", "Greenhouse", "Personio",
        "Adzuna", "Monsterboard", "LinkedIn", "Indeed"
    ])
    company = st.text_input("Enter company domain (e.g. 'openai', 'crowe-foederer')")
    api_key_input = st.text_input("API key (only for Adzuna: app_id:app_key)", type="password")
    if st.button("🔄 Fetch Live Vacancies"):
        df_live = fetch_live_vacancies(src, company, api_key_input)
        if df_live.empty:
            st.warning("No vacancies found or connection unavailable.")
        else:
            st.success(f"✅ {len(df_live)} vacancies fetched from {src}")
            st.dataframe(
                df_live[[c for c in ["title", "location", "url"] if c in df_live.columns]].head(50),
                use_container_width=True
            )
            selected = st.selectbox("Select vacancy to attach", df_live["title"].tolist())
            if st.button("📎 Attach & Run AI Analysis"):
                desc = df_live.loc[df_live["title"] == selected, "description"].values[0]
                st.session_state["vacancy_select"] = selected
                st.toast(f"Vacancy '{selected}' attached — analyzing...", icon="🧠")
                run_full_analysis(desc)
    st.markdown('</div>', unsafe_allow_html=True)

# ===== FUNCTIONAL DOCUMENTATION =====
with section[8]:
    with st.expander("📘 Functional Documentation / Functionele Documentatie", expanded=False):
        st.markdown(f"""
        ## 🇬🇧 How to Use the Personato TalentLens App

        **Overview:**  
        The *Personato TalentLens* platform uses AI and predictive analytics to assess and match candidates with job vacancies.  
        It combines document parsing, predictive modeling, and conversational intelligence to provide end-to-end recruitment insights.

        ### Step 1: Dashboard
        - Displays the overall candidate dataset and model performance.  
        - View KPIs: total candidates, average success rate, and culture fit.  
        - The histogram shows distribution of motivation scores.  
        - When you analyze a CV, its prediction appears here along with a “Feature Snapshot” bar chart.  
        - The **Matched Vacancy Badge** shows which sector and role were auto-detected from your uploaded vacancy text.

        ### Step 2: Conversational Recruiter
        - Upload one or multiple **CVs** and optional **cover letters** (PDF, DOCX, TXT).  
        - Upload or paste a **job vacancy** on the left sidebar — the AI will automatically detect its sector and closest matching job title.  
        - After uploading, the app combines both texts to perform:
          - Feature extraction (skills, motivation, emotion, experience)
          - Success and acceptance probability prediction  
          - Narrative generation and grounded Q&A via ChatGPT (EN/NL bilingual)
        - Use the **thread system** to keep conversations organized; you can archive or delete threads at any time.
        - Use the **Batch Narratives** option to instantly generate summaries for all uploaded CVs.

        ### Step 3: Explorer
        - Displays candidate patterns (Experience vs Skill Match) and a live sample of the dataset.  
        - Shows candidate-specific features after analysis.  
        - Use this to visually interpret hiring trends and identify outliers.

        ### Step 4: Assessment & Insights
        - Evaluate reasoning, motivation, and role fit for each candidate.  
        - Automatically generates dynamic questions based on the vacancy, CV, and cover letter.  
        - Compute **Assessment Fit**, **Motivation**, and **Emotion Scores**.  
        - Generates radar charts and AI-based narrative summaries.  
        - Export PDF/HTML assessment reports directly.

        ### Step 5: Bias & Explainability
        - Visualizes **global SHAP values** to explain model behavior.  
        - Displays **sector baseline SHAP** analysis — e.g., how skill or motivation influences hiring differently by sector.  
        - Useful for fairness audits and model interpretability.

        ### Step 6: Settings & Data
        - Shows active vacancy definitions (from built-in sectors or uploaded CSV).  
        - Displays sample training data for transparency.  
        - Adjust **Custom Factors** for each sector (Experience, Skill Match, Culture Fit, Motivation, Sentiment, Emotion).  
        - Use the **Blend Slider** to control how much custom weights influence predictions.  
        - The **Personalized Outreach** generator creates tailored candidate messages from the latest analysis.

        ### Step 7: Vacancy Auto-Match (Sidebar)
        - Paste or upload a job vacancy — the AI reads it and automatically matches it to the most similar sector and job title.  
        - A green toast notification confirms the detection with a similarity score.  
        - You can still override the selection manually using the dropdowns below.

        ### Step 8: OCR for Scanned PDFs
        - If you upload scanned (image-based) PDFs, enable **“OCR for scanned PDFs”** in the sidebar.  
        - Adjust **DPI** (300 recommended) and **PSM mode** (4, 11, or 12 for multi-column or low-quality scans).  
        - The system will re-extract text using Tesseract and Poppler.

        ### Step 9: Conversational AI (Deep Analysis)
        - Ask direct questions such as:
          - “Summarize the candidate’s strongest skills.”  
          - “Which parts of the CV best demonstrate leadership?”  
          - “How does this candidate compare to a Senior Data Engineer role?”  
        - The AI references only your uploaded documents for factual answers.

        ### Step 10: Batch Analysis & Export
        - Upload multiple CVs to get a CSV of narratives and feature metrics (skill match, motivation, success, etc.).  
        - Results can be downloaded instantly for HR review or dashboard integration.

        ---

        ## 🇳🇱 Hoe gebruik je de Personato TalentLens-app

        **Overzicht:**  
        *Personato TalentLens* combineert AI, tekstanalyse en voorspellende modellen om kandidaten beter te koppelen aan vacatures.  
        De tool ondersteunt recruiters bij het beoordelen van motivatie, cultuurfit en kans op succes.

        ### Stap 1: Dashboard
        - Toont algemene model- en datasetstatistieken.  
        - Bekijk KPI’s: totaal aantal kandidaten, gemiddelde succeskans en cultuurfit.  
        - Na een analyse verschijnt de voorspelling met een **Feature Snapshot**-grafiek.  
        - De **Matched Vacancy Badge** laat zien welke sector en functie automatisch zijn herkend uit de vacaturetekst.

        ### Stap 2: Conversational Recruiter
        - Upload één of meerdere **CV’s** en eventueel een **motivatiebrief**.  
        - Upload of plak een **vacaturetekst** in de zijbalk — de AI herkent automatisch de juiste sector en functietitel.  
        - De analyse genereert:
          - Functie- en vaardigheidsscores  
          - Succeskans en verwachte acceptatie  
          - AI-gegenereerde narratieven en antwoorden op basis van het CV en de vacature  
        - Gebruik het **gesprekssysteem** om analyses te bewaren, archiveren of verwijderen.  
        - Met **Batch Narratives** maak je in één keer samenvattingen voor alle geüploade CV’s.

        ### Stap 3: Verkenner
        - Visualiseert patronen tussen ervaring en vaardigheden.  
        - Toont voorbeelddata en het profiel van de laatst geanalyseerde kandidaat.  

        ### Stap 4: Beoordeling & Inzichten
        - Genereert automatisch dynamische vragen op basis van de vacature, het CV en de motivatiebrief.  
        - Bereken **Beoordelingsscore**, **Motivatie** en **Emotiescores**.  
        - Genereer radarplots en AI-samenvattingen in PDF/HTML-formaat.

        ### Stap 5: Bias & Uitlegbaarheid
        - Toont SHAP-plots om te begrijpen waarom het model een bepaalde voorspelling maakt.  
        - Geeft sector-specifieke inzichten (bijvoorbeeld: in de IT-sector is ‘SkillMatch’ belangrijker dan ‘Motivatie’).  

        ### Stap 6: Instellingen & Data
        - Pas **Custom Factors** per sector aan.  
        - Gebruik de **Blend-schuif** om te bepalen in welke mate eigen wegingen de voorspelling beïnvloeden.  
        - Genereer direct een **persoonlijk bericht** voor de kandidaat op basis van de laatste analyse.

        ### Stap 7: Vacature Auto-Match
        - Plak of upload een vacature, waarna de AI automatisch de sector en rol herkent.  
        - De detectie wordt bevestigd met een melding en overeenkomstscore.  
        - Je kunt de selectie altijd handmatig aanpassen.

        ### Stap 8: OCR voor gescande PDF’s
        - Activeer **“OCR voor gescande PDF’s”** bij het uploaden van niet-doorzoekbare bestanden.  
        - Stel **DPI** en **PSM-modus** in voor betere herkenning.  

        ### Stap 9: Gespreks-AI
        - Stel vragen als:
          - “Vat de belangrijkste vaardigheden samen.”  
          - “Wat zijn mogelijke risico’s voor deze kandidaat?”  
          - “Hoe goed past deze kandidaat bij de functie Data Engineer?”  
        - De AI gebruikt uitsluitend geüploade documenten om antwoorden te genereren.

        ### Stap 10: Batch-analyse & Export
        - Analyseer meerdere CV’s tegelijk.  
        - Download alle resultaten als CSV met succeskans, motivatie en fit-scores.

        ---

        **End of Documentation**
        """)





















