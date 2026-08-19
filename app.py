"""
Wahala AI — Full Application
==============================
Features:
  - Guest: symptom selection + live diagnosis
  - Registered: PDF reports + diagnosis history
  - Email/password auth with bcrypt hashing
  - SQLite database via SQLAlchemy
  - Severity-weighted ML model
  - Nigerian Hospital Finder (static database)
"""

import os, re, tempfile
from datetime import datetime

os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"]      = os.environ.get("KAGGLE_KEY", "")

import bcrypt, gradio as gr, kagglehub, numpy as np, pandas as pd
from pathlib import Path
from sqlalchemy import (Column, DateTime, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)

# ══════════════════════════════════════════════════════════════════════════════
# 1. HOSPITAL DATABASE
# ══════════════════════════════════════════════════════════════════════════════
HOSPITALS = [
    # ── Lagos ─────────────────────────────────────────────────────────────────
    {"name":"Lagos University Teaching Hospital (LUTH)","state":"Lagos",
     "address":"Idi-Araba, Surulere, Lagos","phone":"+234-1-774-0082",
     "type":"Teaching","emergency":True},
    {"name":"Lagos Island General Hospital","state":"Lagos",
     "address":"1 Broad St, Lagos Island, Lagos","phone":"+234-1-266-0469",
     "type":"General","emergency":True},
    {"name":"Reddington Hospital","state":"Lagos",
     "address":"12 Idowu Martins St, Victoria Island, Lagos",
     "phone":"+234-1-280-9750","type":"Private","emergency":True},
    {"name":"Eko Hospital","state":"Lagos",
     "address":"31 Mobolaji Bank Anthony Way, Ikeja, Lagos",
     "phone":"+234-1-493-6000","type":"Private","emergency":True},
    {"name":"General Hospital Lagos","state":"Lagos",
     "address":"1 Marina, Lagos Island, Lagos","phone":"+234-1-263-0683",
     "type":"General","emergency":True},
    {"name":"St. Nicholas Hospital","state":"Lagos",
     "address":"57 Campbell St, Lagos Island, Lagos","phone":"+234-1-263-5032",
     "type":"Private","emergency":False},
    {"name":"Havana Specialist Hospital","state":"Lagos",
     "address":"22 Simbiat Abiola Way, Ikeja, Lagos","phone":"+234-1-493-1122",
     "type":"Private","emergency":False},
    {"name":"Apapa General Hospital","state":"Lagos",
     "address":"1 Hospital Rd, Apapa, Lagos","phone":"+234-1-587-0206",
     "type":"General","emergency":False},

    # ── Abuja (FCT) ───────────────────────────────────────────────────────────
    {"name":"National Hospital Abuja","state":"FCT Abuja",
     "address":"Plot 132 Central Business District, Abuja",
     "phone":"+234-9-461-4500","type":"Teaching","emergency":True},
    {"name":"University of Abuja Teaching Hospital","state":"FCT Abuja",
     "address":"Gwagwalada, Abuja","phone":"+234-9-882-1280",
     "type":"Teaching","emergency":True},
    {"name":"Garki Hospital Abuja","state":"FCT Abuja",
     "address":"Area 3, Garki, Abuja","phone":"+234-9-234-6171",
     "type":"General","emergency":True},
    {"name":"Nisa Premier Hospital","state":"FCT Abuja",
     "address":"3 DamIbrahim Babangida Way, Jabi, Abuja",
     "phone":"+234-9-291-0364","type":"Private","emergency":True},
    {"name":"Cedarcrest Hospital","state":"FCT Abuja",
     "address":"1 Tafawa Balewa Way, Area 11, Abuja",
     "phone":"+234-9-291-5522","type":"Private","emergency":False},

    # ── Kano ──────────────────────────────────────────────────────────────────
    {"name":"Aminu Kano Teaching Hospital","state":"Kano",
     "address":"Zaria Rd, Kano","phone":"+234-64-666-291",
     "type":"Teaching","emergency":True},
    {"name":"Muhammad Abdullahi Wase Specialist Hospital","state":"Kano",
     "address":"Zaria Road, Kano","phone":"+234-64-630-280",
     "type":"General","emergency":True},
    {"name":"Murtala Muhammad Specialist Hospital","state":"Kano",
     "address":"Club Road, Kano","phone":"+234-64-643-551",
     "type":"General","emergency":True},

    # ── Rivers ────────────────────────────────────────────────────────────────
    {"name":"University of Port Harcourt Teaching Hospital","state":"Rivers",
     "address":"East-West Rd, Choba, Port Harcourt",
     "phone":"+234-84-230-641","type":"Teaching","emergency":True},
    {"name":"Braithwaite Memorial Specialist Hospital","state":"Rivers",
     "address":"1 Braithwaite Memorial Hospital Rd, Port Harcourt",
     "phone":"+234-84-238-740","type":"General","emergency":True},
    {"name":"Meridian Hospital","state":"Rivers",
     "address":"Trans Amadi, Port Harcourt","phone":"+234-84-462-200",
     "type":"Private","emergency":True},

    # ── Oyo ───────────────────────────────────────────────────────────────────
    {"name":"University College Hospital (UCH)","state":"Oyo",
     "address":"Queen Elizabeth Rd, Ibadan","phone":"+234-2-241-1768",
     "type":"Teaching","emergency":True},
    {"name":"Adeoyo State Hospital","state":"Oyo",
     "address":"Ring Road, Ibadan","phone":"+234-2-231-0923",
     "type":"General","emergency":True},
    {"name":"Jericho Nursing Home","state":"Oyo",
     "address":"Jericho GRA, Ibadan","phone":"+234-2-241-0088",
     "type":"Private","emergency":False},

    # ── Enugu ─────────────────────────────────────────────────────────────────
    {"name":"University of Nigeria Teaching Hospital","state":"Enugu",
     "address":"Ituku-Ozalla, Enugu","phone":"+234-42-253-225",
     "type":"Teaching","emergency":True},
    {"name":"Enugu State University Teaching Hospital","state":"Enugu",
     "address":"Park Lane, GRA, Enugu","phone":"+234-42-259-321",
     "type":"Teaching","emergency":True},
    {"name":"Park Lane General Hospital","state":"Enugu",
     "address":"1 Park Lane, GRA, Enugu","phone":"+234-42-256-188",
     "type":"General","emergency":True},

    # ── Delta ─────────────────────────────────────────────────────────────────
    {"name":"Delta State University Teaching Hospital","state":"Delta",
     "address":"Oghara, Delta State","phone":"+234-53-680-012",
     "type":"Teaching","emergency":True},
    {"name":"Central Hospital Warri","state":"Delta",
     "address":"Warri, Delta State","phone":"+234-53-255-823",
     "type":"General","emergency":True},

    # ── Edo ───────────────────────────────────────────────────────────────────
    {"name":"University of Benin Teaching Hospital","state":"Edo",
     "address":"PMB 1111, Benin City","phone":"+234-52-600-355",
     "type":"Teaching","emergency":True},
    {"name":"Central Hospital Benin","state":"Edo",
     "address":"Airport Rd, Benin City","phone":"+234-52-255-764",
     "type":"General","emergency":True},
    {"name":"Irrua Specialist Teaching Hospital","state":"Edo",
     "address":"Irrua, Edo State","phone":"+234-57-200-150",
     "type":"Teaching","emergency":True},

    # ── Kaduna ────────────────────────────────────────────────────────────────
    {"name":"Ahmadu Bello University Teaching Hospital","state":"Kaduna",
     "address":"Shika, Zaria, Kaduna","phone":"+234-69-550-571",
     "type":"Teaching","emergency":True},
    {"name":"Barau Dikko Teaching Hospital","state":"Kaduna",
     "address":"Kawo, Kaduna","phone":"+234-62-244-011",
     "type":"Teaching","emergency":True},
    {"name":"St. Gerard Catholic Hospital","state":"Kaduna",
     "address":"8 Ahmadu Bello Way, Kaduna","phone":"+234-62-240-011",
     "type":"Private","emergency":False},

    # ── Anambra ───────────────────────────────────────────────────────────────
    {"name":"Nnamdi Azikiwe University Teaching Hospital","state":"Anambra",
     "address":"Nnewi, Anambra State","phone":"+234-46-462-131",
     "type":"Teaching","emergency":True},
    {"name":"Chukwuemeka Odumegwu Ojukwu University Teaching Hospital",
     "state":"Anambra","address":"Awka, Anambra State",
     "phone":"+234-48-550-443","type":"Teaching","emergency":True},

    # ── Imo ───────────────────────────────────────────────────────────────────
    {"name":"Imo State University Teaching Hospital","state":"Imo",
     "address":"Orlu Road, Owerri","phone":"+234-83-230-420",
     "type":"Teaching","emergency":True},
    {"name":"Federal Medical Centre Owerri","state":"Imo",
     "address":"Owerri, Imo State","phone":"+234-83-231-551",
     "type":"General","emergency":True},

    # ── Ogun ──────────────────────────────────────────────────────────────────
    {"name":"Olabisi Onabanjo University Teaching Hospital","state":"Ogun",
     "address":"Sagamu, Ogun State","phone":"+234-37-640-585",
     "type":"Teaching","emergency":True},
    {"name":"Federal Medical Centre Abeokuta","state":"Ogun",
     "address":"Idi-Aba, Abeokuta","phone":"+234-39-241-201",
     "type":"General","emergency":True},

    # ── Osun ──────────────────────────────────────────────────────────────────
    {"name":"Obafemi Awolowo University Teaching Hospital","state":"Osun",
     "address":"Ile-Ife, Osun State","phone":"+234-36-230-374",
     "type":"Teaching","emergency":True},
    {"name":"LAUTECH Teaching Hospital","state":"Osun",
     "address":"Osogbo, Osun State","phone":"+234-35-242-271",
     "type":"Teaching","emergency":True},

    # ── Cross River ───────────────────────────────────────────────────────────
    {"name":"University of Calabar Teaching Hospital","state":"Cross River",
     "address":"Moore Rd, Calabar","phone":"+234-87-232-940",
     "type":"Teaching","emergency":True},
    {"name":"General Hospital Calabar","state":"Cross River",
     "address":"Mayne Ave, Calabar","phone":"+234-87-232-255",
     "type":"General","emergency":True},

    # ── Borno ─────────────────────────────────────────────────────────────────
    {"name":"University of Maiduguri Teaching Hospital","state":"Borno",
     "address":"Maiduguri, Borno State","phone":"+234-76-232-505",
     "type":"Teaching","emergency":True},
    {"name":"Umaru Shehu Ultra-Modern Hospital","state":"Borno",
     "address":"Maiduguri, Borno State","phone":"+234-76-234-100",
     "type":"General","emergency":True},

    # ── Plateau ───────────────────────────────────────────────────────────────
    {"name":"Jos University Teaching Hospital","state":"Plateau",
     "address":"Tafawa Balewa Way, Jos","phone":"+234-73-452-600",
     "type":"Teaching","emergency":True},
    {"name":"Bingham University Teaching Hospital","state":"Plateau",
     "address":"Jos, Plateau State","phone":"+234-73-460-212",
     "type":"Teaching","emergency":True},

    # ── Kwara ─────────────────────────────────────────────────────────────────
    {"name":"University of Ilorin Teaching Hospital","state":"Kwara",
     "address":"Ilorin, Kwara State","phone":"+234-31-221-924",
     "type":"Teaching","emergency":True},
    {"name":"General Hospital Ilorin","state":"Kwara",
     "address":"Hospital Rd, Ilorin","phone":"+234-31-220-631",
     "type":"General","emergency":True},

    # ── Sokoto ────────────────────────────────────────────────────────────────
    {"name":"Usmanu Danfodiyo University Teaching Hospital","state":"Sokoto",
     "address":"Sokoto, Sokoto State","phone":"+234-60-232-240",
     "type":"Teaching","emergency":True},
    {"name":"Specialist Hospital Sokoto","state":"Sokoto",
     "address":"Sokoto, Sokoto State","phone":"+234-60-231-870",
     "type":"General","emergency":True},

    # ── Bauchi ────────────────────────────────────────────────────────────────
    {"name":"Abubakar Tafawa Balewa University Teaching Hospital",
     "state":"Bauchi","address":"Bauchi, Bauchi State",
     "phone":"+234-77-543-200","type":"Teaching","emergency":True},

    # ── Niger ─────────────────────────────────────────────────────────────────
    {"name":"IBB Specialist Hospital","state":"Niger",
     "address":"Minna, Niger State","phone":"+234-66-222-640",
     "type":"General","emergency":True},
    {"name":"Federal Medical Centre Bida","state":"Niger",
     "address":"Bida, Niger State","phone":"+234-66-461-370",
     "type":"General","emergency":True},

    # ── Akwa Ibom ─────────────────────────────────────────────────────────────
    {"name":"University of Uyo Teaching Hospital","state":"Akwa Ibom",
     "address":"Uyo, Akwa Ibom State","phone":"+234-85-200-640",
     "type":"Teaching","emergency":True},
    {"name":"Ibom Specialist Hospital","state":"Akwa Ibom",
     "address":"Uyo, Akwa Ibom State","phone":"+234-85-203-781",
     "type":"General","emergency":True},

    # ── Abia ──────────────────────────────────────────────────────────────────
    {"name":"Federal Medical Centre Umuahia","state":"Abia",
     "address":"Umuahia, Abia State","phone":"+234-88-220-640",
     "type":"General","emergency":True},
    {"name":"Abia State University Teaching Hospital","state":"Abia",
     "address":"Aba, Abia State","phone":"+234-82-225-640",
     "type":"Teaching","emergency":True},
]

ALL_STATES = sorted(set(h["state"] for h in HOSPITALS))


def get_hospitals(state: str, sev_score: int, top_n: int = 6):
    """
    Returns hospitals filtered by state, sorted by relevance to severity.
    CRITICAL/HIGH  -> teaching and emergency hospitals first
    MODERATE       -> general hospitals first
    LOW            -> any hospital
    """
    filtered = [h for h in HOSPITALS if h["state"] == state]
    if not filtered:
        return []

    def rank(h):
        score = 0
        if sev_score >= 45 and h["emergency"]:
            score += 10
        if sev_score >= 70 and h["type"] == "Teaching":
            score += 5
        if h["type"] == "Teaching":
            score += 3
        elif h["type"] == "General":
            score += 2
        elif h["type"] == "Private":
            score += 1
        return score

    return sorted(filtered, key=rank, reverse=True)[:top_n]


def render_hospital_cards(hospitals: list, sev_score: int,
                           sev_lbl: str, sev_col: str, compact=False):
    """Renders hospital cards as HTML."""
    if not hospitals:
        return """
        <div style='text-align:center;padding:2rem;color:#94a3b8;
                    font-family:system-ui;'>
            <p>No hospitals found for this state yet.</p>
            <p style='font-size:0.8rem;'>
                More states coming soon — contact us to add your area.
            </p>
        </div>"""

    type_colors = {
        "Teaching": ("#0d9488","#f0fdf4"),
        "General" : ("#2563eb","#eff6ff"),
        "Private" : ("#a855f7","#faf5ff"),
        "Clinic"  : ("#d97706","#fffbeb"),
    }

    cards = ""
    for h in hospitals:
        tc, bg = type_colors.get(h["type"], ("#64748b","#f8fafc"))
        emergency_badge = (
            "<span style='background:#fef2f2;color:#dc2626;padding:2px 6px;"
            "border-radius:999px;font-size:0.68rem;font-weight:700;"
            "margin-left:6px;'>🚨 EMERGENCY</span>"
            if h["emergency"] else ""
        )
        padding = "0.8rem" if compact else "1.1rem"
        cards += f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;
                    border-radius:12px;padding:{padding};margin-bottom:0.7rem;
                    box-shadow:0 1px 3px rgba(0,0,0,0.05);
                    border-left:4px solid {tc};'>
            <div style='display:flex;justify-content:space-between;
                        align-items:flex-start;flex-wrap:wrap;gap:4px;'>
                <div style='flex:1;'>
                    <div style='display:flex;align-items:center;
                                flex-wrap:wrap;gap:4px;margin-bottom:4px;'>
                        <strong style='color:#1e293b;font-size:0.88rem;
                                       font-family:system-ui;'>{h["name"]}</strong>
                        {emergency_badge}
                    </div>
                    <p style='margin:0;font-size:0.78rem;color:#475569;'>
                        📍 {h["address"]}
                    </p>
                    <p style='margin:4px 0 0 0;font-size:0.78rem;color:#475569;'>
                        📞 <a href='tel:{h["phone"]}'
                              style='color:#0d9488;text-decoration:none;'>
                            {h["phone"]}
                        </a>
                    </p>
                </div>
                <span style='background:{bg};color:{tc};padding:3px 10px;
                             border-radius:999px;font-size:0.72rem;
                             font-weight:700;white-space:nowrap;'>
                    {h["type"]}
                </span>
            </div>
        </div>"""
    return cards


def hospital_finder_html(state: str, sev_score: int):
    """Full hospital finder panel for the Hospitals tab."""
    if not state:
        return """
        <div style='text-align:center;padding:2rem;color:#94a3b8;
                    font-family:system-ui;'>
            <div style='font-size:2.5rem;'>🏥</div>
            <p>Select your state to find nearby hospitals.</p>
        </div>"""

    sev_lbl, sev_col, sev_bg = severity_label(sev_score)
    hospitals = get_hospitals(state, sev_score, top_n=8)

    advisory = {
        "CRITICAL": ("🚨 CRITICAL — Go to the nearest emergency room immediately. "
                     "Call for an ambulance if possible. Do not delay."),
        "HIGH"    : ("⚠️ HIGH severity — Visit a Teaching or General Hospital "
                     "today. Do not self-medicate."),
        "MODERATE": ("📋 MODERATE severity — Schedule an appointment at a General "
                     "Hospital or clinic within 1-2 days."),
        "LOW"     : ("✅ LOW severity — A visit to a nearby clinic or general "
                     "practitioner should suffice."),
    }.get(sev_lbl, "")

    advisory_bg  = {"CRITICAL":"#fef2f2","HIGH":"#fffbeb",
                    "MODERATE":"#eff6ff","LOW":"#f0fdf4"}.get(sev_lbl,"#f8fafc")
    advisory_col = {"CRITICAL":"#dc2626","HIGH":"#d97706",
                    "MODERATE":"#2563eb","LOW":"#16a34a"}.get(sev_lbl,"#64748b")

    cards = render_hospital_cards(hospitals, sev_score, sev_lbl, sev_col)

    return f"""
    <div style='font-family:system-ui;'>
        <div style='background:{advisory_bg};border:1px solid {advisory_col}33;
                    border-radius:12px;padding:0.9rem 1rem;margin-bottom:1rem;'>
            <p style='margin:0;font-size:0.85rem;color:{advisory_col};
                      font-weight:600;line-height:1.5;'>{advisory}</p>
        </div>
        <p style='font-size:0.73rem;color:#64748b;margin:0 0 10px 0;
                  font-weight:700;text-transform:uppercase;
                  letter-spacing:0.05em;'>
            Recommended Hospitals in {state}
            <span style='font-weight:400;'>(sorted by severity match)</span>
        </p>
        {cards}
        <p style='font-size:0.72rem;color:#94a3b8;margin-top:1rem;
                  text-align:center;'>
            📞 Always call ahead to confirm availability before visiting.
        </p>
    </div>"""


def quick_hospitals_html(state: str, sev_score: int):
    """Compact top-3 hospital preview shown at bottom of diagnosis results."""
    if not state or sev_score == 0:
        return ""
    sev_lbl, sev_col, _ = severity_label(sev_score)
    hospitals = get_hospitals(state, sev_score, top_n=3)
    if not hospitals:
        return ""
    cards = render_hospital_cards(hospitals, sev_score,
                                   sev_lbl, sev_col, compact=True)
    return f"""
    <div style='margin-top:1.2rem;'>
        <p style='font-size:0.73rem;color:#64748b;margin:0 0 8px 0;
                  font-weight:700;text-transform:uppercase;
                  letter-spacing:0.05em;'>
            🏥 Nearby Hospitals — {state}
        </p>
        {cards}
        <p style='font-size:0.72rem;color:#94a3b8;margin-top:0.5rem;'>
            See the 🏥 Hospitals tab for more options.
        </p>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════
Base    = declarative_base()
engine  = create_engine("sqlite:///wahala_ai.db",
                         connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True)
    email         = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name     = Column(String, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    diagnoses     = relationship("DiagnosisRecord", back_populates="user",
                                  cascade="all, delete-orphan")


class DiagnosisRecord(Base):
    __tablename__  = "diagnoses"
    id             = Column(Integer, primary_key=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    symptoms       = Column(Text,    nullable=False)
    top_disease    = Column(String,  nullable=False)
    probability    = Column(String,  nullable=False)
    severity_score = Column(Integer, nullable=False)
    age            = Column(Integer, nullable=False)
    gender         = Column(String,  nullable=False)
    duration       = Column(String,  nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    user           = relationship("User", back_populates="diagnoses")


Base.metadata.create_all(engine)


# ══════════════════════════════════════════════════════════════════════════════
# 3. AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hash_password(plain):
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain, hashed):
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def valid_email(email):
    return bool(re.match(r"^[\w.\-]+@[\w.\-]+\.\w{2,}$", email))

def register_user(full_name, email, password, confirm):
    if not full_name.strip():
        return False, "Full name is required.", None
    if not valid_email(email):
        return False, "Please enter a valid email address.", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters.", None
    if password != confirm:
        return False, "Passwords do not match.", None
    db = Session()
    try:
        if db.query(User).filter_by(email=email.lower()).first():
            return False, "An account with this email already exists.", None
        user = User(email=email.lower(),
                    password_hash=hash_password(password),
                    full_name=full_name.strip())
        db.add(user); db.commit(); db.refresh(user)
        return True, f"Welcome to Wahala AI, {full_name.split()[0]}! 🎉", user.id
    finally:
        db.close()

def login_user(email, password):
    if not email or not password:
        return False, "Please enter your email and password.", None, None
    db = Session()
    try:
        user = db.query(User).filter_by(email=email.lower()).first()
        if not user or not verify_password(password, user.password_hash):
            return False, "Incorrect email or password.", None, None
        return True, f"Welcome back, {user.full_name.split()[0]}! 👋", user.id, user.full_name
    finally:
        db.close()

def save_diagnosis(user_id, symptoms, top_disease, probability,
                   severity_score, age, gender, duration):
    db = Session()
    try:
        db.add(DiagnosisRecord(
            user_id=user_id, symptoms=", ".join(symptoms),
            top_disease=top_disease, probability=probability,
            severity_score=severity_score, age=age,
            gender=gender, duration=duration))
        db.commit()
    finally:
        db.close()

def get_user_history(user_id):
    db = Session()
    try:
        rows = (db.query(DiagnosisRecord)
                  .filter_by(user_id=user_id)
                  .order_by(DiagnosisRecord.created_at.desc())
                  .limit(20).all())
        return [{"date"       : r.created_at.strftime("%b %d, %Y %I:%M %p"),
                 "top_disease": r.top_disease,
                 "probability": r.probability,
                 "severity"   : r.severity_score,
                 "symptoms"   : r.symptoms,
                 "age"        : r.age,
                 "gender"     : r.gender,
                 "duration"   : r.duration} for r in rows]
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("⬇️  Loading datasets...")
path  = kagglehub.dataset_download(
    "kaushil268/disease-prediction-using-machine-learning")
path2 = Path(kagglehub.dataset_download(
    "itachi9604/disease-symptom-description-dataset"))

df            = pd.read_csv(f"{path}/Training.csv")
df_desc       = pd.read_csv(path2 / "symptom_Description.csv")
df_precaution = pd.read_csv(path2 / "symptom_precaution.csv")
df_severity   = pd.read_csv(path2 / "Symptom-severity.csv")

df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
             errors="ignore").dropna(axis=1, how="all")

target       = "prognosis"
symptom_cols = [c for c in df.columns if c != target]
df[symptom_cols] = df[symptom_cols].astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 5. SEVERITY MAP
# ══════════════════════════════════════════════════════════════════════════════
df_severity.columns    = df_severity.columns.str.strip().str.lower()
df_severity["symptom"] = (df_severity["symptom"].str.strip()
                           .str.lower().str.replace(" ", "_"))
df_severity["weight"]  = pd.to_numeric(df_severity["weight"],
                                        errors="coerce").fillna(1)
severity_map = dict(zip(df_severity["symptom"], df_severity["weight"]))

def get_severity_score(sel):
    if not sel: return 0, []
    w = [severity_map.get(s, 1) for s in sel]
    return (round((sum(w) / (7*len(sel)))*100),
            sorted([(s, severity_map.get(s,1)) for s in sel],
                    key=lambda x: x[1], reverse=True))

def severity_label(score):
    if score >= 70: return "CRITICAL","#dc2626","#fef2f2"
    if score >= 45: return "HIGH",    "#d97706","#fffbeb"
    if score >= 25: return "MODERATE","#2563eb","#eff6ff"
    return              "LOW",    "#16a34a","#f0fdf4"

# ══════════════════════════════════════════════════════════════════════════════
# 6. LOOKUP DICTS
# ══════════════════════════════════════════════════════════════════════════════
df_desc.columns = df_desc.columns.str.strip()
df_precaution.columns = df_precaution.columns.str.strip()
desc_dict = dict(zip(df_desc["Disease"].str.strip(), df_desc.iloc[:, 1]))
prec_dict = {}
for _, row in df_precaution.iterrows():
    d = str(row["Disease"]).strip()
    prec_dict[d] = [str(row[f"Precaution_{i}"]).strip()
                    for i in range(1,5) if pd.notna(row.get(f"Precaution_{i}"))]

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("⏳ Training model...")
X_weighted = df[symptom_cols].copy()
for col in symptom_cols:
    X_weighted[col] *= severity_map.get(col, 1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X_weighted, y, test_size=0.2, random_state=42, stratify=y)
rf    = RandomForestClassifier(n_estimators=300, max_depth=20,
                                class_weight="balanced",
                                random_state=42, n_jobs=-1)
model = CalibratedClassifierCV(rf, method="isotonic", cv=5)
model.fit(X_train, y_train)
print(f"✅ Model ready — Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. PATIENT CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
def apply_patient_context(results, age, gender):
    age_risk = {
        "Heart attack"        :{"elder":1.4,"adult":1.1},
        "Hypertension "       :{"elder":1.3,"adult":1.1},
        "Diabetes "           :{"elder":1.2,"adult":1.1},
        "Arthritis"           :{"elder":1.3},
        "Pneumonia"           :{"elder":1.3,"child":1.2},
        "Bronchial Asthma"    :{"child":1.2},
        "Common Cold"         :{"child":1.2},
        "Malaria"             :{"child":1.1},
        "Tuberculosis"        :{"adult":1.1},
        "Cervical spondylosis":{"elder":1.3,"adult":1.1},
    }
    gender_risk = {
        "Heart attack"           :{"Male":1.2},
        "Hypertension "          :{"Male":1.1},
        "Arthritis"              :{"Female":1.2},
        "Osteoarthristis"        :{"Female":1.2},
        "Hypothyroidism"         :{"Female":1.3},
        "Hyperthyroidism"        :{"Female":1.3},
        "Urinary tract infection":{"Female":1.4},
    }
    ag = "child" if age < 18 else "elder" if age >= 60 else "adult"
    adj = []
    for d, p in results:
        f = 1.0
        if d in age_risk    and ag     in age_risk[d]:    f *= age_risk[d][ag]
        if d in gender_risk and gender in gender_risk[d]: f *= gender_risk[d][gender]
        adj.append((d, p*f))
    t = sum(p for _,p in adj)
    return [(d, p/t) for d,p in adj]

# ══════════════════════════════════════════════════════════════════════════════
# 9. PDF GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(patient_name, age, gender, duration, state,
                 selected_symptoms, top_results, sev_score):
    tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf",
                                       prefix="WahalaAI_")
    doc  = SimpleDocTemplate(tmp.name, pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm,  bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []
    teal   = colors.HexColor("#0d9488")
    dark   = colors.HexColor("#1e293b")
    gray   = colors.HexColor("#64748b")

    # Header
    ht = Table([[
        Paragraph("<font color='#0d9488' size=22><b>⚕ Wahala AI</b></font>",
                  styles["Normal"]),
        Paragraph(f"<font color='#64748b' size=8>Diagnostic Report<br/>"
                  f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
                  f"</font>", ParagraphStyle("r", alignment=2))
    ]], colWidths=[10*cm, 7*cm])
    ht.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                             ("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story += [ht, HRFlowable(width="100%",thickness=2,color=teal,spaceAfter=12)]

    # Patient info
    sev_lbl,_,_ = severity_label(sev_score)
    story.append(Paragraph("<b>PATIENT INFORMATION</b>",
                           ParagraphStyle("s",fontSize=9,textColor=gray,spaceAfter=6)))
    pt = Table([
        ["Full Name", patient_name or "Not provided",
         "Severity",  f"{sev_score}/100 ({sev_lbl})"],
        ["Age",       f"{age} years","Gender",gender],
        ["Duration",  duration,"State",state or "Not specified"],
    ], colWidths=[3.5*cm,5.5*cm,3.5*cm,4.5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#f8fafc")),
        ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0,0),(0,-1),gray), ("TEXTCOLOR",(2,0),(2,-1),gray),
        ("FONTNAME",  (0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",  (2,0),(2,-1),"Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1),9),
        ("GRID",      (0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),
        ("PADDING",   (0,0),(-1,-1),6),
    ]))
    story += [pt, Spacer(1,14)]

    # Symptoms
    story.append(Paragraph("<b>SYMPTOMS REPORTED</b>",
                           ParagraphStyle("s",fontSize=9,textColor=gray,spaceAfter=6)))
    story.append(Paragraph(" · ".join([s.replace("_"," ").title()
                                        for s in selected_symptoms]),
                           ParagraphStyle("sym",fontSize=9,textColor=dark,
                                          leading=14,spaceAfter=14)))

    # Severity table
    story.append(Paragraph("<b>SYMPTOM SEVERITY ANALYSIS</b>",
                           ParagraphStyle("s",fontSize=9,textColor=gray,spaceAfter=6)))
    _,breakdown = get_severity_score(selected_symptoms)
    sev_data = [["Symptom","Weight","Level"]]
    for sym,w in breakdown:
        sev_data.append([sym.replace("_"," ").title(), f"{w}/7",
                         "Critical" if w>=6 else "High" if w>=4
                         else "Moderate" if w>=2 else "Low"])
    st = Table(sev_data, colWidths=[9*cm,3*cm,5*cm])
    st.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),teal),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
        ("PADDING",(0,0),(-1,-1),5),
    ]))
    story += [st, Spacer(1,14)]

    # Diagnosis table
    story.append(Paragraph("<b>DIAGNOSIS RESULTS</b>",
                           ParagraphStyle("s",fontSize=9,textColor=gray,spaceAfter=6)))
    diag_data = [["Rank","Disease","Probability","Confidence"]]
    for i,(d,p) in enumerate(top_results):
        pct = p*100
        diag_data.append([f"#{i+1}",d,f"{pct:.1f}%",
                           "HIGH" if pct>=50 else "MODERATE" if pct>=20 else "LOW"])
    dt = Table(diag_data, colWidths=[2*cm,9*cm,3.5*cm,2.5*cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),teal),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
        ("PADDING",(0,0),(-1,-1),5),
        ("ALIGN",(2,0),(3,-1),"CENTER"),
        ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
        ("TEXTCOLOR",(0,1),(0,-1),teal),
    ]))
    story += [dt, Spacer(1,14)]

    # Top disease detail
    if top_results:
        td = top_results[0][0]
        story.append(Paragraph("<b>TOP DIAGNOSIS DETAIL</b>",
                               ParagraphStyle("s",fontSize=9,textColor=gray,spaceAfter=6)))
        story.append(Paragraph(f"<b>{td}</b>",
                               ParagraphStyle("dn",fontSize=11,textColor=teal,spaceAfter=4)))
        story.append(Paragraph(desc_dict.get(td,"No description available."),
                               ParagraphStyle("desc",fontSize=9,textColor=dark,
                                              leading=13,spaceAfter=8)))
        precs = prec_dict.get(td,[])
        if precs:
            story.append(Paragraph("<b>Recommended Precautions:</b>",
                                   ParagraphStyle("ph",fontSize=9,textColor=dark,spaceAfter=4)))
            for p in precs:
                story.append(Paragraph(f"• {p.capitalize()}",
                                       ParagraphStyle("pr",fontSize=9,textColor=dark,
                                                      leading=14,leftIndent=12)))
        story.append(Spacer(1,14))

    # Hospital section in PDF
    if state:
        story.append(Paragraph("<b>RECOMMENDED HOSPITALS</b>",
                               ParagraphStyle("s",fontSize=9,textColor=gray,spaceAfter=6)))
        hospitals = get_hospitals(state, sev_score, top_n=5)
        hosp_data = [["Hospital","Type","Phone","Emergency"]]
        for h in hospitals:
            hosp_data.append([h["name"], h["type"], h["phone"],
                               "Yes" if h["emergency"] else "No"])
        ht2 = Table(hosp_data, colWidths=[7*cm,2.5*cm,4*cm,2*cm])
        ht2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),teal),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),7.5),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white,colors.HexColor("#f8fafc")]),
            ("PADDING",(0,0),(-1,-1),4),
            ("ALIGN",(3,0),(3,-1),"CENTER"),
        ]))
        story += [ht2, Spacer(1,14)]

    # Footer
    story += [
        HRFlowable(width="100%",thickness=1,
                   color=colors.HexColor("#e2e8f0"),spaceAfter=8),
        Paragraph("<b>DISCLAIMER:</b> This report is generated by Wahala AI "
                  "for educational and demonstration purposes only. It does "
                  "not constitute medical advice, diagnosis, or treatment. "
                  "Always consult a qualified healthcare professional.",
                  ParagraphStyle("disc",fontSize=7.5,textColor=gray,leading=11)),
        Spacer(1,6),
        Paragraph("Wahala AI · Calibrated Random Forest · Built for African Healthcare 🌍",
                  ParagraphStyle("foot",fontSize=7,
                                 textColor=colors.HexColor("#94a3b8"),
                                 alignment=TA_CENTER)),
    ]
    doc.build(story)
    return tmp.name

# ══════════════════════════════════════════════════════════════════════════════
# 10. DIAGNOSIS HTML
# ══════════════════════════════════════════════════════════════════════════════
def diagnose_html(selected_symptoms, age, gender, duration, state):
    if not selected_symptoms:
        return """
        <div style='text-align:center;padding:3rem;color:#94a3b8;
                    font-family:system-ui;'>
            <div style='font-size:3rem;'>🩺</div>
            <p style='font-size:1rem;margin-top:0.5rem;'>
                Select symptoms on the left to see your diagnosis
            </p>
        </div>"""

    vec   = [severity_map.get(s,1) if s in selected_symptoms else 0
             for s in symptom_cols]
    proba = model.predict_proba([vec])[0]
    raw   = sorted(zip(model.classes_, proba), key=lambda x:x[1], reverse=True)
    adj   = sorted(apply_patient_context(raw, int(age), gender),
                   key=lambda x:x[1], reverse=True)
    top   = adj[:4]
    other = max(0, 1.0 - sum(p for _,p in top))

    sev_score, breakdown     = get_severity_score(selected_symptoms)
    sev_lbl, sev_col, sev_bg = severity_label(sev_score)
    card_colors = ["#0d9488","#a855f7","#f59e0b","#3b82f6"]
    icons       = ["🥇","🥈","🥉","4️⃣"]

    dur_note = {"Less than 1 day":"⚡ Very recent onset",
                "1-3 days":"📅 Short duration","4-7 days":"📆 About a week",
                "1-2 weeks":"🗓 Subacute","More than 2 weeks":"⏳ Chronic"
               }.get(duration,"")

    sym_tags = "".join([
        f"<span style='background:#f0fdf4;color:#15803d;"
        f"border:1px solid #bbf7d0;border-radius:999px;"
        f"padding:3px 10px;font-size:0.75rem;margin:2px;"
        f"display:inline-block;'>✓ {s.replace('_',' ')}</span>"
        for s in selected_symptoms])

    sev_rows = ""
    for sym,w in breakdown[:6]:
        bar   = min(100,int((w/7)*100))
        color = ("#dc2626" if w>=6 else "#d97706" if w>=4
                 else "#2563eb" if w>=2 else "#16a34a")
        sev_rows += f"""
        <tr>
          <td style='padding:5px 8px;font-size:0.78rem;color:#1e293b;
                     font-weight:500;'>{sym.replace('_',' ').title()}</td>
          <td style='padding:5px 8px;width:130px;'>
            <div style='background:#e2e8f0;border-radius:999px;height:7px;'>
              <div style='background:{color};width:{bar}%;height:100%;
                          border-radius:999px;'></div>
            </div>
          </td>
          <td style='padding:5px 8px;font-size:0.78rem;color:{color};
                     font-weight:700;text-align:right;'>{w}/7</td>
        </tr>"""

    cards = ""
    for i,(disease,prob) in enumerate(top):
        pct = prob*100
        bw  = min(100,int(pct))
        col = card_colors[i]
        badge = (
            "<span style='background:#fef2f2;color:#dc2626;padding:2px 8px;"
            "border-radius:999px;font-size:0.7rem;font-weight:700;'>HIGH</span>"
            if pct>=50 else
            "<span style='background:#fffbeb;color:#d97706;padding:2px 8px;"
            "border-radius:999px;font-size:0.7rem;font-weight:700;'>MODERATE</span>"
            if pct>=20 else
            "<span style='background:#f0fdf4;color:#16a34a;padding:2px 8px;"
            "border-radius:999px;font-size:0.7rem;font-weight:700;'>LOW</span>"
        )
        desc  = desc_dict.get(disease,"No description available.")
        precs = prec_dict.get(disease,[])
        prec_html = ""
        if precs:
            items = "".join([
                f"<li style='margin-bottom:6px;color:#1e293b;font-size:0.8rem;"
                f"font-weight:500;line-height:1.4;'>{p.capitalize()}</li>"
                for p in precs])
            prec_html = f"""
            <div style='margin-top:0.8rem;background:#f8fafc;
                        border:1px solid #e2e8f0;border-left:3px solid {col};
                        border-radius:10px;padding:0.8rem 1rem;'>
                <p style='font-size:0.75rem;font-weight:700;color:{col};
                          margin:0 0 8px 0;text-transform:uppercase;
                          letter-spacing:0.04em;'>🛡️ Recommended Precautions</p>
                <ul style='margin:0;padding-left:1.2rem;list-style:disc;'>
                    {items}</ul>
            </div>"""
        cards += f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;
                    border-radius:14px;padding:1.2rem;margin-bottom:0.9rem;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
            <div style='display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:0.5rem;'>
                <div style='display:flex;align-items:center;gap:8px;'>
                    <span style='font-size:1.1rem;'>{icons[i]}</span>
                    <strong style='color:#1e293b;font-size:0.95rem;
                                   font-family:system-ui;'>{disease}</strong>
                </div>
                <div style='display:flex;align-items:center;gap:6px;'>
                    {badge}
                    <span style='background:#f1f5f9;color:#1e293b;
                                 padding:3px 10px;border-radius:999px;
                                 font-size:0.82rem;font-weight:700;'>
                        {pct:.1f}%</span>
                </div>
            </div>
            <div style='background:#f1f5f9;border-radius:999px;height:8px;'>
                <div style='background:{col};width:{bw}%;height:100%;
                            border-radius:999px;'></div>
            </div>
            <p style='font-size:0.8rem;color:#374151;margin:0.7rem 0 0.3rem 0;
                      line-height:1.6;border-left:3px solid {col};
                      padding-left:8px;'>{desc}</p>
            {prec_html}
        </div>"""

    if other > 0.001:
        cards += f"""
        <div style='background:#f8fafc;border:1px dashed #cbd5e1;
                    border-radius:14px;padding:1rem;margin-bottom:0.8rem;'>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#64748b;font-size:0.85rem;'>
                    🔍 Other possibilities combined</span>
                <span style='color:#64748b;font-size:0.85rem;font-weight:700;'>
                    {other*100:.1f}%</span>
            </div>
        </div>"""

    warning = ""
    if top[0][1] < 0.35:
        warning = """
        <div style='background:#fef2f2;border:1px solid #fecaca;
                    border-radius:10px;padding:0.8rem 1rem;margin-top:0.5rem;'>
            <p style='margin:0;color:#dc2626;font-size:0.82rem;'>
                ⚠️ <strong>Low confidence</strong> — add more symptoms or
                consult a healthcare professional.</p>
        </div>"""

    hosp_section = quick_hospitals_html(state, sev_score)

    return f"""
    <div style='font-family:system-ui;'>
        <div style='background:#f8fafc;border:1px solid #e2e8f0;
                    border-radius:12px;padding:0.8rem 1rem;margin-bottom:1rem;
                    font-size:0.82rem;color:#475569;'>
            👤 <strong style='color:#1e293b;'>Patient:</strong>
            {int(age)} yrs · {gender} · {dur_note}
            {f"· 📍 {state}" if state else ""}
        </div>
        <div style='background:{sev_bg};border:1px solid {sev_col}44;
                    border-radius:12px;padding:1rem;margin-bottom:1rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <p style='margin:0;font-size:0.73rem;color:{sev_col};
                              font-weight:700;text-transform:uppercase;
                              letter-spacing:0.05em;'>Overall Severity Score</p>
                    <p style='margin:4px 0 0 0;font-size:1.7rem;
                              font-weight:800;color:{sev_col};'>{sev_score}
                        <span style='font-size:0.85rem;font-weight:400;'>/100</span>
                    </p>
                </div>
                <span style='background:{sev_col};color:white;padding:5px 16px;
                             border-radius:999px;font-size:0.82rem;
                             font-weight:700;'>{sev_lbl}</span>
            </div>
            <table style='width:100%;border-collapse:collapse;margin-top:0.8rem;'>
                {sev_rows}
            </table>
        </div>
        <div style='margin-bottom:1rem;'>
            <p style='font-size:0.73rem;color:#64748b;margin:0 0 6px 0;
                      font-weight:700;text-transform:uppercase;
                      letter-spacing:0.05em;'>Active Symptoms</p>
            <div>{sym_tags}</div>
        </div>
        <p style='font-size:0.73rem;color:#64748b;margin:0 0 10px 0;
                  font-weight:700;text-transform:uppercase;
                  letter-spacing:0.05em;'>Diagnosis Results</p>
        {cards}
        {warning}
        {hosp_section}
        <p style='font-size:0.72rem;color:#94a3b8;margin-top:1.2rem;
                  text-align:center;'>
            ⚕️ Wahala AI · Educational prototype ·
            Not a substitute for professional medical advice
        </p>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# 11. HISTORY HTML
# ══════════════════════════════════════════════════════════════════════════════
def render_history(user_id):
    if not user_id:
        return "<p style='color:#94a3b8;text-align:center;padding:2rem;'>Log in to see your history.</p>"
    records = get_user_history(user_id)
    if not records:
        return "<p style='color:#94a3b8;text-align:center;padding:2rem;'>No diagnoses yet. Run your first diagnosis!</p>"
    cards = ""
    for r in records:
        sl,sc,sb = severity_label(r["severity"])
        cards += f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;
                    border-radius:12px;padding:1rem;margin-bottom:0.8rem;
                    box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display:flex;justify-content:space-between;
                        align-items:flex-start;'>
                <div>
                    <p style='margin:0;font-size:0.72rem;color:#94a3b8;'>
                        {r["date"]}</p>
                    <p style='margin:4px 0;font-size:1rem;font-weight:700;
                              color:#1e293b;'>{r["top_disease"]}</p>
                    <p style='margin:0;font-size:0.8rem;color:#475569;'>
                        {r["age"]} yrs · {r["gender"]} · {r["duration"]}</p>
                </div>
                <div style='text-align:right;'>
                    <span style='font-size:1.2rem;font-weight:800;
                                 color:#0d9488;'>{r["probability"]}</span><br/>
                    <span style='background:{sb};color:{sc};padding:2px 8px;
                                 border-radius:999px;font-size:0.7rem;
                                 font-weight:700;'>{sl}</span>
                </div>
            </div>
            <p style='margin:8px 0 0 0;font-size:0.75rem;color:#64748b;
                      line-height:1.5;'>
                <strong>Symptoms:</strong> {r["symptoms"]}</p>
        </div>"""
    return f"""
    <div style='font-family:system-ui;'>
        <p style='font-size:0.73rem;color:#64748b;margin:0 0 10px 0;
                  font-weight:700;text-transform:uppercase;
                  letter-spacing:0.05em;'>
            Your Last {len(records)} Diagnoses</p>
        {cards}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# 12. GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
css = """
.gradio-container{max-width:1100px !important;margin:0 auto !important;}
footer{display:none !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Wahala AI") as demo:

    user_id_state   = gr.State(None)
    user_name_state = gr.State(None)
    sev_score_state = gr.State(0)

    gr.HTML("""
    <div style='background:linear-gradient(135deg,#0d9488,#0f766e);
                border-radius:16px;padding:2rem;margin-bottom:1rem;
                color:white;font-family:system-ui;'>
        <div style='display:flex;align-items:center;gap:14px;'>
            <span style='font-size:2.8rem;'>⚕️</span>
            <div>
                <h1 style='margin:0;font-size:2.2rem;font-weight:800;
                           letter-spacing:-0.02em;'>Wahala AI</h1>
                <p style='margin:4px 0 0 0;opacity:0.85;font-size:0.88rem;'>
                    Severity-weighted diagnosis · 41 diseases · 132 symptoms
                    · Hospital Finder · PDF Reports
                </p>
            </div>
        </div>
    </div>""")

    gr.HTML("""
    <div style='background:#fffbeb;border:1px solid #fde68a;border-radius:12px;
                padding:0.9rem 1.2rem;margin-bottom:1.2rem;font-family:system-ui;
                display:flex;gap:10px;align-items:flex-start;'>
        <span style='font-size:1.3rem;'>⚠️</span>
        <p style='margin:0;font-size:0.83rem;color:#92400e;line-height:1.5;'>
            <strong>Educational Prototype Only.</strong> Wahala AI is built for
            research and demonstration. It does not provide real medical advice.
            Always consult a qualified healthcare professional.
        </p>
    </div>""")

    user_status_bar = gr.HTML("""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                padding:0.6rem 1rem;margin-bottom:1rem;font-family:system-ui;
                font-size:0.82rem;color:#64748b;display:flex;
                justify-content:space-between;align-items:center;'>
        <span>👤 Browsing as <strong>Guest</strong></span>
        <span style='font-size:0.75rem;color:#0d9488;'>
            Log in to unlock PDF reports and diagnosis history
        </span>
    </div>""")

    with gr.Tabs():

        # ── TAB 1: Diagnosis ──────────────────────────────────────────────────
        with gr.Tab("🩺 Diagnosis"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 1. Patient Information")
                    with gr.Row():
                        age_input    = gr.Slider(1,100,value=30,step=1,label="Age")
                        gender_input = gr.Radio(["Male","Female"],
                                                 value="Male",label="Gender")
                    duration_input = gr.Dropdown(
                        choices=["Less than 1 day","1-3 days","4-7 days",
                                 "1-2 weeks","More than 2 weeks"],
                        value="1-3 days", label="Symptom Duration")
                    state_input = gr.Dropdown(
                        choices=[""] + ALL_STATES,
                        value="", label="Your State (for hospital finder)",
                        info="Select your state to see nearby hospitals")

                    gr.Markdown("### 2. Select Symptoms")
                    symptom_input = gr.Dropdown(
                        choices=sorted(symptom_cols), multiselect=True,
                        label="Search and select symptoms",
                        info="Type to search — e.g. 'fever', 'cough', 'headache'")
                    with gr.Row():
                        clear_btn  = gr.Button("🗑 Clear All",   variant="secondary")
                        sample_btn = gr.Button("🎲 Load Sample", variant="secondary")

                    gr.Markdown("### 3. PDF Report *(Login required)*")
                    report_btn  = gr.Button("📄 Generate PDF Report",
                                             variant="primary")
                    report_file = gr.File(label="Download Report")
                    report_msg  = gr.HTML("")

                with gr.Column(scale=3):
                    gr.Markdown("### 4. Live Diagnosis")
                    diagnosis_output = gr.HTML(
                        value=diagnose_html([], 30, "Male", "1-3 days", ""))

        # ── TAB 2: Hospital Finder ────────────────────────────────────────────
        with gr.Tab("🏥 Hospital Finder"):
            gr.HTML("""
            <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:12px;padding:0.9rem 1.2rem;
                        margin-bottom:1rem;font-family:system-ui;'>
                <p style='margin:0;font-size:0.83rem;color:#15803d;'>
                    🏥 <strong>Hospital Finder</strong> — recommendations are
                    sorted by your diagnosis severity score. Run a diagnosis
                    first, then select your state here for the best matches.
                </p>
            </div>""")
            with gr.Row():
                hosp_state_input = gr.Dropdown(
                    choices=ALL_STATES, label="Select Your State",
                    info="Choose your state to find nearby hospitals")
                hosp_refresh_btn = gr.Button("🔍 Find Hospitals",
                                              variant="primary", scale=0)
            hospital_output = gr.HTML("""
            <div style='text-align:center;padding:3rem;color:#94a3b8;
                        font-family:system-ui;'>
                <div style='font-size:3rem;'>🏥</div>
                <p>Select your state above and click Find Hospitals.</p>
            </div>""")

        # ── TAB 3: History ────────────────────────────────────────────────────
        with gr.Tab("📊 History *(Login required)*"):
            refresh_btn    = gr.Button("🔄 Refresh History", variant="secondary")
            history_output = gr.HTML(
                "<p style='color:#94a3b8;text-align:center;padding:2rem;'>"
                "Log in to see your diagnosis history.</p>")

        # ── TAB 4: Login ──────────────────────────────────────────────────────
        with gr.Tab("🔐 Login"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Sign In")
                    login_email    = gr.Textbox(label="Email",
                                                placeholder="you@example.com")
                    login_password = gr.Textbox(label="Password",
                                                type="password",
                                                placeholder="Your password")
                    login_btn = gr.Button("Sign In", variant="primary")
                    login_msg = gr.HTML("")
                with gr.Column():
                    gr.Markdown("### Create Account")
                    reg_name     = gr.Textbox(label="Full Name",
                                              placeholder="e.g. Emeka Okafor")
                    reg_email    = gr.Textbox(label="Email",
                                              placeholder="you@example.com")
                    reg_password = gr.Textbox(label="Password", type="password",
                                              placeholder="Min. 6 characters")
                    reg_confirm  = gr.Textbox(label="Confirm Password",
                                              type="password",
                                              placeholder="Repeat password")
                    reg_btn = gr.Button("Create Account", variant="primary")
                    reg_msg = gr.HTML("")

    # ══════════════════════════════════════════════════════════════════════════
    # 13. EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════
    diag_inputs = [symptom_input, age_input, gender_input,
                   duration_input, state_input]

    def update_diagnosis(symptoms, age, gender, duration, state):
        sev,_ = get_severity_score(symptoms) if symptoms else (0,[])
        return diagnose_html(symptoms, age, gender, duration, state), sev

    for comp in diag_inputs:
        comp.change(fn=update_diagnosis, inputs=diag_inputs,
                    outputs=[diagnosis_output, sev_score_state])

    clear_btn.click(
        fn=lambda: ([], diagnose_html([], 30, "Male", "1-3 days", ""), 0),
        outputs=[symptom_input, diagnosis_output, sev_score_state])

    sample_btn.click(
        fn=lambda: (
            ["high_fever","chills","headache","nausea","vomiting"],
            diagnose_html(["high_fever","chills","headache","nausea","vomiting"],
                          30,"Male","1-3 days",""),
            get_severity_score(["high_fever","chills","headache",
                                "nausea","vomiting"])[0]
        ),
        outputs=[symptom_input, diagnosis_output, sev_score_state])

    # Hospital finder
    hosp_refresh_btn.click(
        fn=lambda state, sev: hospital_finder_html(state, sev),
        inputs=[hosp_state_input, sev_score_state],
        outputs=hospital_output)

    # PDF report
    def handle_report(user_id, user_name, symptoms, age, gender,
                      duration, state):
        if not user_id:
            return None, """
            <div style='background:#fef2f2;border:1px solid #fecaca;
                        border-radius:8px;padding:0.7rem 1rem;
                        font-family:system-ui;'>
                <p style='margin:0;color:#dc2626;font-size:0.82rem;'>
                    🔐 Please log in to generate PDF reports.</p>
            </div>"""
        if not symptoms:
            return None, """
            <div style='background:#fffbeb;border:1px solid #fde68a;
                        border-radius:8px;padding:0.7rem 1rem;
                        font-family:system-ui;'>
                <p style='margin:0;color:#92400e;font-size:0.82rem;'>
                    ⚠️ Please select symptoms first.</p>
            </div>"""
        vec   = [severity_map.get(s,1) if s in symptoms else 0
                 for s in symptom_cols]
        proba = model.predict_proba([vec])[0]
        raw   = sorted(zip(model.classes_, proba),
                       key=lambda x:x[1], reverse=True)
        top   = sorted(apply_patient_context(raw, int(age), gender),
                       key=lambda x:x[1], reverse=True)[:5]
        sev_score,_ = get_severity_score(symptoms)
        save_diagnosis(user_id, symptoms, top[0][0],
                       f"{top[0][1]*100:.1f}%", sev_score,
                       age, gender, duration)
        pdf = generate_pdf(user_name, age, gender, duration, state,
                           symptoms, top, sev_score)
        return pdf, """
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                    border-radius:8px;padding:0.7rem 1rem;
                    font-family:system-ui;'>
            <p style='margin:0;color:#15803d;font-size:0.82rem;'>
                ✅ Report generated and saved to your history!</p>
        </div>"""

    report_btn.click(fn=handle_report,
                     inputs=[user_id_state, user_name_state, symptom_input,
                              age_input, gender_input, duration_input,
                              state_input],
                     outputs=[report_file, report_msg])

    refresh_btn.click(fn=lambda uid: render_history(uid),
                      inputs=user_id_state, outputs=history_output)

    def status_bar(name):
        return f"""
        <div style='background:#f8fafc;border:1px solid #e2e8f0;
                    border-radius:10px;padding:0.6rem 1rem;
                    margin-bottom:1rem;font-family:system-ui;
                    font-size:0.82rem;color:#64748b;display:flex;
                    justify-content:space-between;align-items:center;'>
            <span>👤 Logged in as
                <strong style='color:#0d9488;'>{name}</strong></span>
            <span style='font-size:0.75rem;color:#16a34a;font-weight:600;'>
                ✅ Full access unlocked</span>
        </div>"""

    def ok_msg(msg):
        return f"""
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                    border-radius:8px;padding:0.7rem 1rem;'>
            <p style='margin:0;color:#15803d;font-size:0.85rem;'>✅ {msg}</p>
        </div>"""

    def err_msg(msg):
        return f"""
        <div style='background:#fef2f2;border:1px solid #fecaca;
                    border-radius:8px;padding:0.7rem 1rem;'>
            <p style='margin:0;color:#dc2626;font-size:0.85rem;'>❌ {msg}</p>
        </div>"""

    def handle_login(email, password):
        ok, msg, uid, name = login_user(email, password)
        if ok:
            return (uid, name, ok_msg(msg),
                    status_bar(name), render_history(uid))
        return None, None, err_msg(msg), gr.update(), gr.update()

    login_btn.click(fn=handle_login,
                    inputs=[login_email, login_password],
                    outputs=[user_id_state, user_name_state,
                             login_msg, user_status_bar, history_output])

    def handle_register(name, email, password, confirm):
        ok, msg, uid = register_user(name, email, password, confirm)
        if ok:
            return uid, name, ok_msg(msg), status_bar(name)
        return None, None, err_msg(msg), gr.update()

    reg_btn.click(fn=handle_register,
                  inputs=[reg_name, reg_email, reg_password, reg_confirm],
                  outputs=[user_id_state, user_name_state,
                           reg_msg, user_status_bar])

PORT = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=PORT)
