import os, kagglehub, gradio as gr, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
import tempfile

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("⬇️  Loading datasets...")
path  = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
path2 = Path(kagglehub.dataset_download("itachi9604/disease-symptom-description-dataset"))

df            = pd.read_csv(f"{path}/Training.csv")
df_desc       = pd.read_csv(path2 / "symptom_Description.csv")
df_precaution = pd.read_csv(path2 / "symptom_precaution.csv")
df_severity   = pd.read_csv(path2 / "Symptom-severity.csv")

df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors='ignore')
df = df.dropna(axis=1, how='all')

target       = 'prognosis'
symptom_cols = [c for c in df.columns if c != target]
df[symptom_cols] = df[symptom_cols].astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 2. SEVERITY MAP
# ══════════════════════════════════════════════════════════════════════════════
df_severity.columns    = df_severity.columns.str.strip().str.lower()
df_severity["symptom"] = (df_severity["symptom"].str.strip()
                           .str.lower().str.replace(" ", "_"))
df_severity["weight"]  = pd.to_numeric(df_severity["weight"],
                                        errors="coerce").fillna(1)
severity_map = dict(zip(df_severity["symptom"], df_severity["weight"]))

def get_severity_score(selected_symptoms):
    if not selected_symptoms:
        return 0, []
    weights   = [severity_map.get(s, 1) for s in selected_symptoms]
    score     = round((sum(weights) / (7 * len(selected_symptoms))) * 100)
    breakdown = sorted([(s, severity_map.get(s, 1)) for s in selected_symptoms],
                        key=lambda x: x[1], reverse=True)
    return score, breakdown

def severity_label(score):
    if score >= 70: return "CRITICAL", "#dc2626", "#fef2f2"
    if score >= 45: return "HIGH",     "#d97706", "#fffbeb"
    if score >= 25: return "MODERATE", "#2563eb", "#eff6ff"
    return              "LOW",     "#16a34a", "#f0fdf4"

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOOKUP DICTS
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
# 4. TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("⏳ Training model...")
X_weighted = df[symptom_cols].copy()
for col in symptom_cols:
    X_weighted[col] = X_weighted[col] * severity_map.get(col, 1)

y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X_weighted, y, test_size=0.2, random_state=42, stratify=y
)

rf    = RandomForestClassifier(n_estimators=300, max_depth=20,
                                class_weight='balanced',
                                random_state=42, n_jobs=-1)
model = CalibratedClassifierCV(rf, method='isotonic', cv=5)
model.fit(X_train, y_train)
print(f"✅ Model ready — Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PATIENT CONTEXT
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
# 6. PDF GENERATOR
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
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</font>",
            ParagraphStyle("r", alignment=2))
    ]], colWidths=[10*cm, 7*cm])
    ht.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BOTTOMPADDING",(0,0),(-1,-1),8)
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
        ("BACKGROUND",(0,0),(0,-1), colors.HexColor("#f8fafc")),
        ("BACKGROUND",(2,0),(2,-1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0,0),(0,-1), gray),
        ("TEXTCOLOR", (2,0),(2,-1), gray),
        ("FONTNAME",  (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0),(2,-1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1), 9),
        ("GRID",      (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",   (0,0),(-1,-1), 6),
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
        ("BACKGROUND",(0,0),(-1,0), teal),
        ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1), 8.5),
        ("GRID",      (0,0),(-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING",   (0,0),(-1,-1), 5),
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
        ("BACKGROUND",(0,0),(-1,0), teal),
        ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1), 8.5),
        ("GRID",      (0,0),(-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING",   (0,0),(-1,-1), 5),
        ("ALIGN",     (2,0),(3,-1), "CENTER"),
        ("FONTNAME",  (0,1),(0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0,1),(0,-1), teal),
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
# 7. DIAGNOSIS HTML
# ══════════════════════════════════════════════════════════════════════════════
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

    vec   = [severity_map.get(s, 1) if s in selected_symptoms else 0
             for s in symptom_cols]
    proba = model.predict_proba([vec])[0]
    raw   = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)

    adjusted = apply_patient_context(raw, int(age), gender)
    adjusted = sorted(adjusted, key=lambda x: x[1], reverse=True)
    top      = adjusted[:4]
    other    = max(0, 1.0 - sum(p for _, p in top))

    sev_score, breakdown       = get_severity_score(selected_symptoms)
    sev_lbl, sev_col, sev_bg  = severity_label(sev_score)
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
          <td style='padding:5px 8px;font-size:0.78rem;
                     color:#1e293b;font-weight:500;'>
              {sym.replace('_',' ').title()}
          </td>
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
            prec_items = "".join([
                f"<li style='margin-bottom:6px;color:#1e293b;"
                f"font-size:0.8rem;font-weight:500;line-height:1.4;'>"
                f"{p.capitalize()}</li>"
                for p in precs
            ])
            prec_html = f"""
            <div style='margin-top:0.8rem;background:#f8fafc;
                        border:1px solid #e2e8f0;
                        border-left:3px solid {color};
                        border-radius:10px;padding:0.8rem 1rem;'>
                <p style='font-size:0.75rem;font-weight:700;color:{color};
                          margin:0 0 8px 0;text-transform:uppercase;
                          letter-spacing:0.04em;'>
                    🛡️ Recommended Precautions
                </p>
                <ul style='margin:0;padding-left:1.2rem;list-style:disc;'>
                    {prec_items}
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
                <div style='background:{color};width:{bar_width}%;
                            height:100%;border-radius:999px;'></div>
            </div>
            <p style='font-size:0.8rem;color:#374151;
                      margin:0.7rem 0 0.3rem 0;line-height:1.6;
                      border-left:3px solid {color};
                      padding-left:8px;'>{desc}</p>
            {prec_html}
        </div>"""

    if other > 0.001:
        cards += f"""
        <div style='background:#f8fafc;border:1px dashed #cbd5e1;
                    border-radius:14px;padding:1rem;margin-bottom:0.8rem;'>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#64748b;font-size:0.85rem;'>
                    🔍 Other possibilities combined
                </span>
                <span style='color:#64748b;font-size:0.85rem;font-weight:700;'>
                    {other*100:.1f}%
                </span>
            </div>
        </div>"""

    warning = ""
    if top[0][1] < 0.35:
        warning = """
        <div style='background:#fef2f2;border:1px solid #fecaca;
                    border-radius:10px;padding:0.8rem 1rem;margin-top:0.5rem;'>
            <p style='margin:0;color:#dc2626;font-size:0.82rem;'>
                ⚠️ <strong>Low confidence</strong> — add more symptoms or
                consult a healthcare professional.
            </p>
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
                        <span style='font-size:0.85rem;font-weight:400;'>
                            /100
                        </span>
                    </p>
                </div>
                <span style='background:{sev_col};color:white;
                             padding:5px 16px;border-radius:999px;
                             font-size:0.82rem;font-weight:700;'>
                    {sev_lbl}
                </span>
            </div>
            <table style='width:100%;border-collapse:collapse;
                          margin-top:0.8rem;'>
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


def generate_report(patient_name, age, gender, duration, selected_symptoms):
    if not selected_symptoms:
        return None
    vec      = [severity_map.get(s, 1) if s in selected_symptoms else 0
                for s in symptom_cols]
    proba    = model.predict_proba([vec])[0]
    raw      = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)
    adjusted = apply_patient_context(raw, int(age), gender)
    top      = sorted(adjusted, key=lambda x: x[1], reverse=True)[:5]
    sev_score, _ = get_severity_score(selected_symptoms)
    return generate_pdf(patient_name, age, gender, duration,
                        selected_symptoms, top, sev_score)

# ══════════════════════════════════════════════════════════════════════════════
# 8. GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
css = """
.gradio-container { max-width:1100px !important; margin:0 auto !important; }
footer { display:none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Wahala AI") as demo:

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
            medical advice, diagnosis, or treatment. Always consult a
            qualified healthcare professional.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Patient Information")
            patient_name = gr.Textbox(label="Patient Name",
                                       placeholder="e.g. Emeka Okafor")
            with gr.Row():
                age_input    = gr.Slider(1, 100, value=30, step=1, label="Age")
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
            gr.Markdown("### 3. Download Report")
            report_btn  = gr.Button("📄 Generate PDF Report", variant="primary")
            report_file = gr.File(label="Your Report")

        with gr.Column(scale=3):
            gr.Markdown("### 4. Live Diagnosis")
            diagnosis_output = gr.HTML(
                value=diagnose_html([], 30, "Male", "1-3 days")
            )

    diag_inputs = [symptom_input, age_input, gender_input, duration_input]
    for comp in diag_inputs:
        comp.change(fn=diagnose_html, inputs=diag_inputs,
                    outputs=diagnosis_output)

    clear_btn.click(
        fn=lambda: ([], diagnose_html([], 30, "Male", "1-3 days")),
        outputs=[symptom_input, diagnosis_output]
    )
    sample_btn.click(
        fn=lambda: (
            ["high_fever","chills","headache","nausea","vomiting"],
            diagnose_html(["high_fever","chills","headache","nausea","vomiting"],
                          30, "Male", "1-3 days")
        ),
        outputs=[symptom_input, diagnosis_output]
    )
    report_btn.click(
        fn=generate_report,
        inputs=[patient_name, age_input, gender_input,
                duration_input, symptom_input],
        outputs=report_file
    )

PORT = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=PORT)
