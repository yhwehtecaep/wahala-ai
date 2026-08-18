"""
Wahala AI — Full Application
==============================
Features:
  - Guest: symptom selection + live diagnosis
  - Registered: PDF reports + diagnosis history
  - Email/password auth with bcrypt hashing
  - SQLite database via SQLAlchemy
  - Severity-weighted ML model (Random Forest + isotonic calibration)
"""

import os
import re
import tempfile
from datetime import datetime

# ── Kaggle credentials from environment ──────────────────────────────────────
os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"]      = os.environ.get("KAGGLE_KEY", "")

# ── Third-party imports ───────────────────────────────────────────────────────
import bcrypt
import gradio as gr
import kagglehub
import numpy as np
import pandas as pd
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
# 1. DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════
Base      = declarative_base()
engine    = create_engine("sqlite:///wahala_ai.db", connect_args={"check_same_thread": False})
Session   = sessionmaker(bind=engine)


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
    symptoms       = Column(Text, nullable=False)       # comma-separated
    top_disease    = Column(String, nullable=False)
    probability    = Column(String, nullable=False)     # e.g. "73.2%"
    severity_score = Column(Integer, nullable=False)
    age            = Column(Integer, nullable=False)
    gender         = Column(String, nullable=False)
    duration       = Column(String, nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    user           = relationship("User", back_populates="diagnoses")


Base.metadata.create_all(engine)


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w.\-]+@[\w.\-]+\.\w{2,}$", email))


def register_user(full_name: str, email: str, password: str, confirm: str):
    """Returns (success: bool, message: str, user_id: int | None)"""
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
        user = User(
            email=email.lower(),
            password_hash=hash_password(password),
            full_name=full_name.strip()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return True, f"Welcome to Wahala AI, {full_name.split()[0]}! 🎉", user.id
    finally:
        db.close()


def login_user(email: str, password: str):
    """Returns (success: bool, message: str, user_id: int | None, full_name: str | None)"""
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
        record = DiagnosisRecord(
            user_id=user_id,
            symptoms=", ".join(symptoms),
            top_disease=top_disease,
            probability=probability,
            severity_score=severity_score,
            age=age,
            gender=gender,
            duration=duration,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def get_user_history(user_id):
    db = Session()
    try:
        records = (db.query(DiagnosisRecord)
                   .filter_by(user_id=user_id)
                   .order_by(DiagnosisRecord.created_at.desc())
                   .limit(20)
                   .all())
        return [
            {
                "date"         : r.created_at.strftime("%b %d, %Y %I:%M %p"),
                "top_disease"  : r.top_disease,
                "probability"  : r.probability,
                "severity"     : r.severity_score,
                "symptoms"     : r.symptoms,
                "age"          : r.age,
                "gender"       : r.gender,
                "duration"     : r.duration,
            }
            for r in records
        ]
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD DATASETS
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
             errors="ignore")
df = df.dropna(axis=1, how="all")

target       = "prognosis"
symptom_cols = [c for c in df.columns if c != target]
df[symptom_cols] = df[symptom_cols].astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SEVERITY MAP
# ══════════════════════════════════════════════════════════════════════════════
df_severity.columns    = df_severity.columns.str.strip().str.lower()
df_severity["symptom"] = (df_severity["symptom"].str.strip()
                           .str.lower().str.replace(" ", "_"))
df_severity["weight"]  = pd.to_numeric(df_severity["weight"],
                                        errors="coerce").fillna(1)
severity_map = dict(zip(df_severity["symptom"], df_severity["weight"]))


def get_severity_score(selected):
    if not selected:
        return 0, []
    weights   = [severity_map.get(s, 1) for s in selected]
    score     = round((sum(weights) / (7 * len(selected))) * 100)
    breakdown = sorted([(s, severity_map.get(s, 1)) for s in selected],
                        key=lambda x: x[1], reverse=True)
    return score, breakdown


def severity_label(score):
    if score >= 70: return "CRITICAL", "#dc2626", "#fef2f2"
    if score >= 45: return "HIGH",     "#d97706", "#fffbeb"
    if score >= 25: return "MODERATE", "#2563eb", "#eff6ff"
    return              "LOW",     "#16a34a", "#f0fdf4"


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOOKUP DICTS
# ══════════════════════════════════════════════════════════════════════════════
df_desc.columns       = df_desc.columns.str.strip()
df_precaution.columns = df_precaution.columns.str.strip()

desc_dict = dict(zip(df_desc["Disease"].str.strip(), df_desc.iloc[:, 1]))

prec_dict = {}
for _, row in df_precaution.iterrows():
    disease = str(row["Disease"]).strip()
    precs   = [str(row[f"Precaution_{i}"]).strip()
               for i in range(1, 5)
               if pd.notna(row.get(f"Precaution_{i}"))]
    prec_dict[disease] = precs

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("⏳ Training model...")
X_weighted = df[symptom_cols].copy()
for col in symptom_cols:
    X_weighted[col] = X_weighted[col] * severity_map.get(col, 1)

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
# 7. PATIENT CONTEXT MODIFIER
# ══════════════════════════════════════════════════════════════════════════════
def apply_patient_context(results, age, gender):
    age_risk = {
        "Heart attack"         : {"elder": 1.4, "adult": 1.1},
        "Hypertension "        : {"elder": 1.3, "adult": 1.1},
        "Diabetes "            : {"elder": 1.2, "adult": 1.1},
        "Arthritis"            : {"elder": 1.3},
        "Pneumonia"            : {"elder": 1.3, "child": 1.2},
        "Bronchial Asthma"     : {"child": 1.2},
        "Common Cold"          : {"child": 1.2},
        "Malaria"              : {"child": 1.1},
        "Tuberculosis"         : {"adult": 1.1},
        "Cervical spondylosis" : {"elder": 1.3, "adult": 1.1},
    }
    gender_risk = {
        "Heart attack"            : {"Male": 1.2},
        "Hypertension "           : {"Male": 1.1},
        "Arthritis"               : {"Female": 1.2},
        "Osteoarthristis"         : {"Female": 1.2},
        "Hypothyroidism"          : {"Female": 1.3},
        "Hyperthyroidism"         : {"Female": 1.3},
        "Urinary tract infection" : {"Female": 1.4},
    }
    age_group = "child" if age < 18 else "elder" if age >= 60 else "adult"
    adjusted  = []
    for disease, prob in results:
        factor = 1.0
        if disease in age_risk    and age_group in age_risk[disease]:
            factor *= age_risk[disease][age_group]
        if disease in gender_risk and gender in gender_risk[disease]:
            factor *= gender_risk[disease][gender]
        adjusted.append((disease, prob * factor))
    total = sum(p for _, p in adjusted)
    return [(d, p / total) for d, p in adjusted]

# ══════════════════════════════════════════════════════════════════════════════
# 8. PDF GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(patient_name, age, gender, duration,
                 selected_symptoms, top_results, sev_score):
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf",
                                         prefix="WahalaAI_")
    doc    = SimpleDocTemplate(tmp.name, pagesize=A4,
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
        Paragraph(
            f"<font color='#64748b' size=8>Diagnostic Report<br/>"
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
            f"</font>",
            ParagraphStyle("r", alignment=2))
    ]], colWidths=[10*cm, 7*cm])
    ht.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story += [ht, HRFlowable(width="100%", thickness=2,
                              color=teal, spaceAfter=12)]

    # Patient info
    sev_lbl, _, _ = severity_label(sev_score)
    story.append(Paragraph("<b>PATIENT INFORMATION</b>",
                           ParagraphStyle("s", fontSize=9,
                                          textColor=gray, spaceAfter=6)))
    pt = Table([
        ["Full Name", patient_name or "Not provided",
         "Severity",  f"{sev_score}/100 ({sev_lbl})"],
        ["Age",       f"{age} years", "Gender", gender],
        ["Duration",  duration, "Symptoms", str(len(selected_symptoms))],
    ], colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 4.5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,-1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (2,0),(2,-1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR",  (0,0),(0,-1), gray),
        ("TEXTCOLOR",  (2,0),(2,-1), gray),
        ("FONTNAME",   (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",   (2,0),(2,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 9),
        ("GRID",       (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",    (0,0),(-1,-1), 6),
    ]))
    story += [pt, Spacer(1, 14)]

    # Symptoms
    story.append(Paragraph("<b>SYMPTOMS REPORTED</b>",
                           ParagraphStyle("s", fontSize=9,
                                          textColor=gray, spaceAfter=6)))
    story.append(Paragraph(
        " · ".join([s.replace("_"," ").title() for s in selected_symptoms]),
        ParagraphStyle("sym", fontSize=9, textColor=dark,
                       leading=14, spaceAfter=14)
    ))

    # Severity table
    story.append(Paragraph("<b>SYMPTOM SEVERITY ANALYSIS</b>",
                           ParagraphStyle("s", fontSize=9,
                                          textColor=gray, spaceAfter=6)))
    _, breakdown = get_severity_score(selected_symptoms)
    sev_data = [["Symptom", "Weight", "Level"]]
    for sym, w in breakdown:
        level = ("Critical" if w >= 6 else "High" if w >= 4
                 else "Moderate" if w >= 2 else "Low")
        sev_data.append([sym.replace("_"," ").title(), f"{w}/7", level])
    st = Table(sev_data, colWidths=[9*cm, 3*cm, 5*cm])
    st.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), teal),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING",       (0,0),(-1,-1), 5),
    ]))
    story += [st, Spacer(1, 14)]

    # Diagnosis table
    story.append(Paragraph("<b>DIAGNOSIS RESULTS</b>",
                           ParagraphStyle("s", fontSize=9,
                                          textColor=gray, spaceAfter=6)))
    diag_data = [["Rank", "Disease", "Probability", "Confidence"]]
    for i, (disease, prob) in enumerate(top_results):
        pct  = prob * 100
        conf = "HIGH" if pct >= 50 else "MODERATE" if pct >= 20 else "LOW"
        diag_data.append([f"#{i+1}", disease, f"{pct:.1f}%", conf])
    dt = Table(diag_data, colWidths=[2*cm, 9*cm, 3.5*cm, 2.5*cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), teal),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING",       (0,0),(-1,-1), 5),
        ("ALIGN",         (2,0),(3,-1), "CENTER"),
        ("FONTNAME",      (0,1),(0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,1),(0,-1), teal),
    ]))
    story += [dt, Spacer(1, 14)]

    # Top disease detail
    if top_results:
        top_disease = top_results[0][0]
        story.append(Paragraph("<b>TOP DIAGNOSIS DETAIL</b>",
                               ParagraphStyle("s", fontSize=9,
                                              textColor=gray, spaceAfter=6)))
        story.append(Paragraph(f"<b>{top_disease}</b>",
                               ParagraphStyle("dn", fontSize=11,
                                              textColor=teal, spaceAfter=4)))
        story.append(Paragraph(
            desc_dict.get(top_disease, "No description available."),
            ParagraphStyle("desc", fontSize=9, textColor=dark,
                           leading=13, spaceAfter=8)
        ))
        precs = prec_dict.get(top_disease, [])
        if precs:
            story.append(Paragraph(
                "<b>Recommended Precautions:</b>",
                ParagraphStyle("ph", fontSize=9, textColor=dark, spaceAfter=4)
            ))
            for p in precs:
                story.append(Paragraph(
                    f"• {p.capitalize()}",
                    ParagraphStyle("pr", fontSize=9, textColor=dark,
                                   leading=14, leftIndent=12)
                ))
        story.append(Spacer(1, 14))

    # Footer
    story += [
        HRFlowable(width="100%", thickness=1,
                   color=colors.HexColor("#e2e8f0"), spaceAfter=8),
        Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by Wahala AI for "
            "educational and demonstration purposes only. It does not "
            "constitute medical advice, diagnosis, or treatment. Always "
            "consult a qualified healthcare professional.",
            ParagraphStyle("disc", fontSize=7.5, textColor=gray, leading=11)
        ),
        Spacer(1, 6),
        Paragraph(
            "Wahala AI · Calibrated Random Forest · Built for African Healthcare 🌍",
            ParagraphStyle("foot", fontSize=7,
                           textColor=colors.HexColor("#94a3b8"),
                           alignment=TA_CENTER)
        ),
    ]
    doc.build(story)
    return tmp.name

# ══════════════════════════════════════════════════════════════════════════════
# 9. DIAGNOSIS HTML
# ══════════════════════════════════════════════════════════════════════════════
def run_diagnosis(selected_symptoms, age, gender):
    """Core inference — returns (top list, sev_score, sev_breakdown)"""
    vec   = [severity_map.get(s, 1) if s in selected_symptoms else 0
             for s in symptom_cols]
    proba = model.predict_proba([vec])[0]
    raw   = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)
    adj   = apply_patient_context(raw, int(age), gender)
    adj   = sorted(adj, key=lambda x: x[1], reverse=True)
    sev_score, breakdown = get_severity_score(selected_symptoms)
    return adj[:4], sev_score, breakdown


def diagnose_html(selected_symptoms, age, gender, duration):
    if not selected_symptoms:
        return """
        <div style='text-align:center;padding:3rem;color:#94a3b8;
                    font-family:system-ui;'>
            <div style='font-size:3rem;'>🩺</div>
            <p style='font-size:1rem;margin-top:0.5rem;'>
                Select symptoms on the left to see your diagnosis
            </p>
        </div>"""

    top, sev_score, breakdown = run_diagnosis(selected_symptoms, age, gender)
    other = max(0, 1.0 - sum(p for _, p in top))

    sev_lbl, sev_col, sev_bg = severity_label(sev_score)
    card_colors = ["#0d9488","#a855f7","#f59e0b","#3b82f6"]
    icons       = ["🥇","🥈","🥉","4️⃣"]

    duration_note = {
        "Less than 1 day"  : "⚡ Very recent onset",
        "1-3 days"         : "📅 Short duration",
        "4-7 days"         : "📆 About a week",
        "1-2 weeks"        : "🗓 Subacute",
        "More than 2 weeks": "⏳ Chronic",
    }.get(duration, "")

    symptom_tags = "".join([
        f"<span style='background:#f0fdf4;color:#15803d;"
        f"border:1px solid #bbf7d0;border-radius:999px;"
        f"padding:3px 10px;font-size:0.75rem;margin:2px;"
        f"display:inline-block;'>✓ {s.replace('_',' ')}</span>"
        for s in selected_symptoms
    ])

    sev_rows = ""
    for sym, w in breakdown[:6]:
        bar   = min(100, int((w/7)*100))
        color = ("#dc2626" if w >= 6 else "#d97706" if w >= 4
                 else "#2563eb" if w >= 2 else "#16a34a")
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
    for i, (disease, prob) in enumerate(top):
        pct       = prob * 100
        bar_width = min(100, int(pct))
        color     = card_colors[i]
        badge = (
            "<span style='background:#fef2f2;color:#dc2626;padding:2px 8px;"
            "border-radius:999px;font-size:0.7rem;font-weight:700;'>HIGH</span>"
            if pct >= 50 else
            "<span style='background:#fffbeb;color:#d97706;padding:2px 8px;"
            "border-radius:999px;font-size:0.7rem;font-weight:700;'>MODERATE</span>"
            if pct >= 20 else
            "<span style='background:#f0fdf4;color:#16a34a;padding:2px 8px;"
            "border-radius:999px;font-size:0.7rem;font-weight:700;'>LOW</span>"
        )
        desc  = desc_dict.get(disease, "No description available.")
        precs = prec_dict.get(disease, [])
        prec_html = ""
        if precs:
            items = "".join([
                f"<li style='margin-bottom:6px;color:#1e293b;font-size:0.8rem;"
                f"font-weight:500;line-height:1.4;'>{p.capitalize()}</li>"
                for p in precs
            ])
            prec_html = f"""
            <div style='margin-top:0.8rem;background:#f8fafc;
                        border:1px solid #e2e8f0;border-left:3px solid {color};
                        border-radius:10px;padding:0.8rem 1rem;'>
                <p style='font-size:0.75rem;font-weight:700;color:{color};
                          margin:0 0 8px 0;text-transform:uppercase;
                          letter-spacing:0.04em;'>🛡️ Recommended Precautions</p>
                <ul style='margin:0;padding-left:1.2rem;list-style:disc;'>
                    {items}
                </ul>
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
                        {pct:.1f}%
                    </span>
                </div>
            </div>
            <div style='background:#f1f5f9;border-radius:999px;height:8px;'>
                <div style='background:{color};width:{bar_width}%;height:100%;
                            border-radius:999px;'></div>
            </div>
            <p style='font-size:0.8rem;color:#374151;margin:0.7rem 0 0.3rem 0;
                      line-height:1.6;border-left:3px solid {color};
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

    return f"""
    <div style='font-family:system-ui;'>
        <div style='background:#f8fafc;border:1px solid #e2e8f0;
                    border-radius:12px;padding:0.8rem 1rem;margin-bottom:1rem;
                    font-size:0.82rem;color:#475569;'>
            👤 <strong style='color:#1e293b;'>Patient:</strong>
            {int(age)} yrs · {gender} · {duration_note}
        </div>
        <div style='background:{sev_bg};border:1px solid {sev_col}44;
                    border-radius:12px;padding:1rem;margin-bottom:1rem;'>
            <div style='display:flex;justify-content:space-between;
                        align-items:center;'>
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
            <div>{symptom_tags}</div>
        </div>
        <p style='font-size:0.73rem;color:#64748b;margin:0 0 10px 0;
                  font-weight:700;text-transform:uppercase;
                  letter-spacing:0.05em;'>Diagnosis Results</p>
        {cards}
        {warning}
        <p style='font-size:0.72rem;color:#94a3b8;margin-top:1.2rem;
                  text-align:center;'>
            ⚕️ Wahala AI · Educational prototype only ·
            Not a substitute for professional medical advice
        </p>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# 10. HISTORY HTML
# ══════════════════════════════════════════════════════════════════════════════
def render_history(user_id):
    if not user_id:
        return "<p style='color:#94a3b8;text-align:center;padding:2rem;'>Log in to see your history.</p>"

    records = get_user_history(user_id)
    if not records:
        return "<p style='color:#94a3b8;text-align:center;padding:2rem;'>No diagnoses yet. Run your first diagnosis!</p>"

    cards = ""
    for r in records:
        sev_lbl, sev_col, sev_bg = severity_label(r["severity"])
        cards += f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;
                    border-radius:12px;padding:1rem;margin-bottom:0.8rem;
                    box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display:flex;justify-content:space-between;
                        align-items:flex-start;'>
                <div>
                    <p style='margin:0;font-size:0.72rem;color:#94a3b8;'>
                        {r["date"]}
                    </p>
                    <p style='margin:4px 0;font-size:1rem;font-weight:700;
                              color:#1e293b;'>{r["top_disease"]}</p>
                    <p style='margin:0;font-size:0.8rem;color:#475569;'>
                        {r["age"]} yrs · {r["gender"]} · {r["duration"]}
                    </p>
                </div>
                <div style='text-align:right;'>
                    <span style='font-size:1.2rem;font-weight:800;
                                 color:#0d9488;'>{r["probability"]}</span>
                    <br/>
                    <span style='background:{sev_bg};color:{sev_col};
                                 padding:2px 8px;border-radius:999px;
                                 font-size:0.7rem;font-weight:700;'>
                        {sev_lbl}
                    </span>
                </div>
            </div>
            <p style='margin:8px 0 0 0;font-size:0.75rem;color:#64748b;
                      line-height:1.5;'>
                <strong>Symptoms:</strong> {r["symptoms"]}
            </p>
        </div>"""

    return f"""
    <div style='font-family:system-ui;'>
        <p style='font-size:0.73rem;color:#64748b;margin:0 0 10px 0;
                  font-weight:700;text-transform:uppercase;
                  letter-spacing:0.05em;'>Your Last {len(records)} Diagnoses</p>
        {cards}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# 11. GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
css = """
.gradio-container { max-width:1100px !important; margin:0 auto !important; }
footer { display:none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Wahala AI") as demo:

    # ── Persistent state ──────────────────────────────────────────────────────
    user_id_state   = gr.State(None)
    user_name_state = gr.State(None)

    # ── Header ─────────────────────────────────────────────────────────────────
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
                    · Patient-aware · PDF Reports
                </p>
            </div>
        </div>
    </div>
    """)

    gr.HTML("""
    <div style='background:#fffbeb;border:1px solid #fde68a;
                border-radius:12px;padding:0.9rem 1.2rem;
                margin-bottom:1.2rem;font-family:system-ui;
                display:flex;gap:10px;align-items:flex-start;'>
        <span style='font-size:1.3rem;'>⚠️</span>
        <p style='margin:0;font-size:0.83rem;color:#92400e;line-height:1.5;'>
            <strong>Educational Prototype Only.</strong> Wahala AI is built
            for research and demonstration purposes. It does not provide real
            medical advice. Always consult a qualified healthcare professional.
        </p>
    </div>
    """)

    # ── User status bar ────────────────────────────────────────────────────────
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

    # ── Main tabs ──────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── TAB 1: Diagnosis ──────────────────────────────────────────────────
        with gr.Tab("🩺 Diagnosis"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 1. Patient Information")
                    with gr.Row():
                        age_input    = gr.Slider(1, 100, value=30, step=1,
                                                  label="Age")
                        gender_input = gr.Radio(["Male","Female"],
                                                 value="Male", label="Gender")
                    duration_input = gr.Dropdown(
                        choices=["Less than 1 day","1-3 days","4-7 days",
                                 "1-2 weeks","More than 2 weeks"],
                        value="1-3 days", label="Symptom Duration"
                    )
                    gr.Markdown("### 2. Select Symptoms")
                    symptom_input = gr.Dropdown(
                        choices=sorted(symptom_cols), multiselect=True,
                        label="Search and select symptoms",
                        info="Type to search — e.g. 'fever', 'cough', 'headache'"
                    )
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
                        value=diagnose_html([], 30, "Male", "1-3 days")
                    )

        # ── TAB 2: History ────────────────────────────────────────────────────
        with gr.Tab("📊 History *(Login required)*"):
            refresh_btn    = gr.Button("🔄 Refresh History", variant="secondary")
            history_output = gr.HTML(
                "<p style='color:#94a3b8;text-align:center;padding:2rem;'>"
                "Log in to see your diagnosis history.</p>"
            )

        # ── TAB 3: Login ──────────────────────────────────────────────────────
        with gr.Tab("🔐 Login"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Sign In")
                    login_email    = gr.Textbox(label="Email",
                                                placeholder="you@example.com")
                    login_password = gr.Textbox(label="Password",
                                                type="password",
                                                placeholder="Your password")
                    login_btn      = gr.Button("Sign In", variant="primary")
                    login_msg      = gr.HTML("")

                with gr.Column():
                    gr.Markdown("### Create Account")
                    reg_name     = gr.Textbox(label="Full Name",
                                              placeholder="e.g. Emeka Okafor")
                    reg_email    = gr.Textbox(label="Email",
                                              placeholder="you@example.com")
                    reg_password = gr.Textbox(label="Password",
                                              type="password",
                                              placeholder="Min. 6 characters")
                    reg_confirm  = gr.Textbox(label="Confirm Password",
                                              type="password",
                                              placeholder="Repeat password")
                    reg_btn      = gr.Button("Create Account", variant="primary")
                    reg_msg      = gr.HTML("")

    # ══════════════════════════════════════════════════════════════════════════
    # 12. EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    # Live diagnosis update
    diag_inputs = [symptom_input, age_input, gender_input, duration_input]
    for comp in diag_inputs:
        comp.change(fn=diagnose_html, inputs=diag_inputs,
                    outputs=diagnosis_output)

    # Clear
    clear_btn.click(
        fn=lambda: ([], diagnose_html([], 30, "Male", "1-3 days")),
        outputs=[symptom_input, diagnosis_output]
    )

    # Sample
    sample_btn.click(
        fn=lambda: (
            ["high_fever","chills","headache","nausea","vomiting"],
            diagnose_html(["high_fever","chills","headache",
                           "nausea","vomiting"], 30, "Male", "1-3 days")
        ),
        outputs=[symptom_input, diagnosis_output]
    )

    # PDF report — only for logged-in users
    def handle_report(user_id, user_name, symptoms, age, gender, duration):
        if not user_id:
            return None, """
            <div style='background:#fef2f2;border:1px solid #fecaca;
                        border-radius:8px;padding:0.7rem 1rem;
                        font-family:system-ui;'>
                <p style='margin:0;color:#dc2626;font-size:0.82rem;'>
                    🔐 Please log in to generate PDF reports.
                </p>
            </div>"""
        if not symptoms:
            return None, """
            <div style='background:#fffbeb;border:1px solid #fde68a;
                        border-radius:8px;padding:0.7rem 1rem;
                        font-family:system-ui;'>
                <p style='margin:0;color:#92400e;font-size:0.82rem;'>
                    ⚠️ Please select symptoms first.
                </p>
            </div>"""

        vec      = [severity_map.get(s, 1) if s in symptoms else 0
                    for s in symptom_cols]
        proba    = model.predict_proba([vec])[0]
        raw      = sorted(zip(model.classes_, proba),
                          key=lambda x: x[1], reverse=True)
        adjusted = apply_patient_context(raw, int(age), gender)
        top      = sorted(adjusted, key=lambda x: x[1], reverse=True)[:5]
        sev_score, _ = get_severity_score(symptoms)

        # Save to history
        save_diagnosis(
            user_id=user_id,
            symptoms=symptoms,
            top_disease=top[0][0],
            probability=f"{top[0][1]*100:.1f}%",
            severity_score=sev_score,
            age=age,
            gender=gender,
            duration=duration
        )

        pdf_path = generate_pdf(user_name, age, gender, duration,
                                 symptoms, top, sev_score)
        return pdf_path, """
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                    border-radius:8px;padding:0.7rem 1rem;
                    font-family:system-ui;'>
            <p style='margin:0;color:#15803d;font-size:0.82rem;'>
                ✅ Report generated and saved to your history!
            </p>
        </div>"""

    report_btn.click(
        fn=handle_report,
        inputs=[user_id_state, user_name_state, symptom_input,
                age_input, gender_input, duration_input],
        outputs=[report_file, report_msg]
    )

    # History refresh
    refresh_btn.click(
        fn=lambda uid: render_history(uid),
        inputs=user_id_state,
        outputs=history_output
    )

    # Login
    def handle_login(email, password):
        success, msg, uid, name = login_user(email, password)
        if success:
            status_html = f"""
            <div style='background:#f8fafc;border:1px solid #e2e8f0;
                        border-radius:10px;padding:0.6rem 1rem;
                        margin-bottom:1rem;font-family:system-ui;
                        font-size:0.82rem;color:#64748b;display:flex;
                        justify-content:space-between;align-items:center;'>
                <span>👤 Logged in as <strong style='color:#0d9488;'>
                    {name}</strong></span>
                <span style='font-size:0.75rem;color:#16a34a;font-weight:600;'>
                    ✅ Full access unlocked
                </span>
            </div>"""
            msg_html = f"""
            <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:8px;padding:0.7rem 1rem;'>
                <p style='margin:0;color:#15803d;font-size:0.85rem;'>
                    ✅ {msg}
                </p>
            </div>"""
            return uid, name, msg_html, status_html, render_history(uid)
        else:
            msg_html = f"""
            <div style='background:#fef2f2;border:1px solid #fecaca;
                        border-radius:8px;padding:0.7rem 1rem;'>
                <p style='margin:0;color:#dc2626;font-size:0.85rem;'>
                    ❌ {msg}
                </p>
            </div>"""
            return None, None, msg_html, gr.update(), gr.update()

    login_btn.click(
        fn=handle_login,
        inputs=[login_email, login_password],
        outputs=[user_id_state, user_name_state,
                 login_msg, user_status_bar, history_output]
    )

    # Register
    def handle_register(name, email, password, confirm):
        success, msg, uid = register_user(name, email, password, confirm)
        if success:
            status_html = f"""
            <div style='background:#f8fafc;border:1px solid #e2e8f0;
                        border-radius:10px;padding:0.6rem 1rem;
                        margin-bottom:1rem;font-family:system-ui;
                        font-size:0.82rem;color:#64748b;display:flex;
                        justify-content:space-between;align-items:center;'>
                <span>👤 Logged in as <strong style='color:#0d9488;'>
                    {name}</strong></span>
                <span style='font-size:0.75rem;color:#16a34a;font-weight:600;'>
                    ✅ Full access unlocked
                </span>
            </div>"""
            msg_html = f"""
            <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:8px;padding:0.7rem 1rem;'>
                <p style='margin:0;color:#15803d;font-size:0.85rem;'>
                    ✅ {msg}
                </p>
            </div>"""
            return uid, name, msg_html, status_html
        else:
            msg_html = f"""
            <div style='background:#fef2f2;border:1px solid #fecaca;
                        border-radius:8px;padding:0.7rem 1rem;'>
                <p style='margin:0;color:#dc2626;font-size:0.85rem;'>
                    ❌ {msg}
                </p>
            </div>"""
            return None, None, msg_html, gr.update()

    reg_btn.click(
        fn=handle_register,
        inputs=[reg_name, reg_email, reg_password, reg_confirm],
        outputs=[user_id_state, user_name_state, reg_msg, user_status_bar]
    )

PORT = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=PORT)
