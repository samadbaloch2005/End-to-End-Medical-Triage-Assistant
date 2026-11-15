"""
TriageCare - Clinical Decision Support dashboard (Streamlit)
New UI inspired by the Salus-style layout: navy sidebar, greeting header,
progress ring, analytics sparkline, scheduled patients, assistant panel.

Run:  py -m streamlit run app.py --server.port 8503
"""
from datetime import datetime
import math

import streamlit as st

st.set_page_config(page_title="TriageCare", page_icon="🩺", layout="wide")

# --------------------------------------------------------------------------
# 1. MODEL HOOK  (connect your real model here)
# --------------------------------------------------------------------------
def predict_triage(p: dict) -> dict:
    """
    Must return:
      {"probs": {"Low": 0-1, "Medium": 0-1, "High": 0-1},
       "drivers": [(feature_label, value_text, weight), ...],   # +ve = raises risk
       "source": "model" | "demo"}

    TODO: replace the demo block below with your trained model call, e.g.
        x = build_feature_row(p)               # same order as training
        proba = model.predict_proba([x])[0]
        info  = explain_patient(x, model, explainer, feature_names)  # src.explain
    """
    # ---- DEMO FALLBACK (transparent rule score, NOT the trained model) ----
    d = []
    def add(label, val, w):
        if w:
            d.append((label, val, w))

    add("SpO2", f"{p['spo2']}%", 3 if p["spo2"] < 90 else 1 if p["spo2"] < 94 else 0)
    add("Systolic BP", f"{p['sbp']} mmHg", 3 if p["sbp"] < 90 else 1 if p["sbp"] < 100 else 0)
    add("Heart rate", f"{p['hr']} bpm", 2 if p["hr"] > 120 else 1 if p["hr"] > 100 else 0)
    add("Respiratory rate", f"{p['rr']}/min", 2 if p["rr"] > 24 else 1 if p["rr"] > 20 else 0)
    add("GCS", str(p["gcs"]), 2 if p["gcs"] <= 13 else 0)
    add("Temperature", f"{p['temp']} °C", 1 if p["temp"] >= 38.5 else 0)
    add("Glucose", f"{p['glu']} mg/dL", 1 if (p["glu"] > 200 or p["glu"] < 70) else 0)
    add("Age", f"{p['age']} y", 1 if p["age"] >= 65 else 0)
    add("Pain score", f"{p['pain']}/10", 1 if p["pain"] >= 8 else 0)
    add("Comorbidity", p["comorb"], 1 if p["comorb"] != "None" else 0)

    score = sum(w for _, _, w in d)
    logits = {"Low": 2.0 - score * 0.9, "Medium": 1.0 - abs(score - 4) * 0.5, "High": score * 0.9 - 3.0}
    z = {k: math.exp(v) for k, v in logits.items()}
    total = sum(z.values())
    probs = {k: v / total for k, v in z.items()}
    d.sort(key=lambda t: -t[2])
    return {"probs": probs, "drivers": d[:6], "source": "demo"}


# --------------------------------------------------------------------------
# 2. STYLE
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root{
  --navy:#0A1F5C; --navy2:#0B2E9E; --blue:#1D4ED8; --blue-soft:#E8EEFF;
  --bg:#F2F5FB; --card:#FFFFFF; --ink:#0F1B3D; --muted:#6B7799; --line:#E4E9F5;
  --low:#16A34A; --med:#D97706; --high:#DC2626;
}
html, body, [class*="css"], .stApp{font-family:'Plus Jakarta Sans',system-ui,sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
header[data-testid="stHeader"], #MainMenu, footer{visibility:hidden;height:0;}
.block-container{padding:1.6rem 2.2rem 3rem;max-width:1400px;}

/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--navy) 0%,var(--navy2) 100%);}
[data-testid="stSidebar"] *{color:#E8EEFF;}
[data-testid="stSidebar"] .stRadio label p{font-size:.95rem;font-weight:500;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:.25rem;}
[data-testid="stSidebar"] .stRadio label{padding:.6rem .8rem;border-radius:12px;width:100%;}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:var(--blue);}
[data-testid="stSidebar"] .stRadio label > div:first-child{display:none;}
[data-testid="stSidebar"] [data-baseweb="select"] > div{background:#fff;}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:var(--ink)!important;}
.brand{display:flex;align-items:center;gap:.6rem;font-size:1.35rem;font-weight:800;margin:.2rem 0 1.4rem;}
.brand i{display:inline-flex;width:34px;height:34px;border-radius:10px;background:var(--blue);
  align-items:center;justify-content:center;font-style:normal;}
.side-note{font-size:.72rem;opacity:.7;line-height:1.4;}
.user-card{display:flex;align-items:center;gap:.7rem;padding:.8rem;border-radius:14px;
  background:rgba(255,255,255,.09);margin-top:1.2rem;}
.avatar{width:36px;height:36px;border-radius:50%;background:#fff;color:var(--navy)!important;
  display:flex;align-items:center;justify-content:center;font-weight:800;}
.user-card small{opacity:.7;display:block;}

/* cards */
.card, div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .form-marker){
  background:var(--card);border:1px solid var(--line);border-radius:18px;
  box-shadow:0 6px 24px rgba(15,27,61,.05);}
.card{padding:1.2rem 1.3rem;margin-bottom:1rem;}
.card h3{margin:0 0 .15rem;font-size:1.05rem;font-weight:700;}
.card .sub{color:var(--muted);font-size:.8rem;margin-bottom:.9rem;}
.card.dark{background:linear-gradient(160deg,#1D4ED8,#0B2E9E);color:#fff;border:none;}
.card.dark .sub{color:#C9D6FF;}

/* topbar */
.hello h1{font-size:1.55rem;font-weight:800;margin:0;}
.hello p{margin:.15rem 0 0;color:var(--muted);font-size:.9rem;}

/* ring */
.ring-wrap{display:flex;align-items:center;gap:1.2rem;}
.ring{--p:75;width:112px;height:112px;border-radius:50%;position:relative;flex:none;
  background:conic-gradient(var(--blue) calc(var(--p)*1%), var(--blue-soft) 0);
  display:flex;align-items:center;justify-content:center;}
.ring::before{content:"";position:absolute;inset:11px;background:#fff;border-radius:50%;}
.ring b{position:relative;font-size:1.5rem;font-weight:800;}
.pill-row{display:flex;gap:.6rem;}
.stat{background:var(--blue-soft);border-radius:12px;padding:.55rem .9rem;min-width:70px;}
.stat b{font-size:1.25rem;display:block;} .stat span{font-size:.72rem;color:var(--muted);}
.stat.on{background:var(--blue);color:#fff;} .stat.on span{color:#DCE5FF;}

/* table */
table.pt{width:100%;border-collapse:collapse;font-size:.86rem;}
table.pt th{text-align:left;color:var(--muted);font-weight:600;font-size:.75rem;padding:.4rem .5rem;}
table.pt td{padding:.7rem .5rem;border-top:1px solid var(--line);}
.pill{display:inline-block;padding:.2rem .65rem;border-radius:999px;font-size:.72rem;font-weight:700;}
.pill.Low{background:#DCFCE7;color:var(--low);} .pill.Medium{background:#FEF3C7;color:var(--med);}
.pill.High{background:#FEE2E2;color:var(--high);} .pill.Wait{background:var(--blue-soft);color:var(--blue);}

/* result */
.result{border-left:6px solid var(--c);padding:1rem 1.2rem;background:#fff;border-radius:14px;
  border:1px solid var(--line);border-left:6px solid var(--c);}
.result .lab{color:var(--muted);font-size:.78rem;} .result .big{font-size:2rem;font-weight:800;color:var(--c);}
.prob{display:flex;align-items:center;gap:.8rem;margin:.5rem 0;font-size:.85rem;}
.prob .n{width:64px;color:var(--muted);} .prob .bar{flex:1;height:10px;background:var(--blue-soft);border-radius:99px;overflow:hidden;}
.prob .bar i{display:block;height:100%;border-radius:99px;} .prob .v{width:52px;text-align:right;font-weight:700;}
.drv{display:flex;align-items:center;gap:.7rem;margin:.45rem 0;font-size:.85rem;}
.drv .n{width:120px;font-weight:600;} .drv .val{width:84px;color:var(--muted);}
.drv .bar{flex:1;height:8px;background:var(--blue-soft);border-radius:99px;}
.drv .bar i{display:block;height:100%;background:var(--high);border-radius:99px;}
.banner{background:#FFF7E6;border:1px solid #F6D28B;color:#7A4B00;padding:.6rem .9rem;border-radius:12px;font-size:.8rem;margin-bottom:.8rem;}
.assist{text-align:center;padding:.4rem 0 .2rem;}
.orb{width:84px;height:84px;margin:.4rem auto .8rem;border-radius:50%;
  background:radial-gradient(circle at 30% 30%,#fff,#9DB7FF 45%,#1D4ED8);box-shadow:0 10px 30px rgba(29,78,216,.35);}
.ds{display:flex;justify-content:space-between;padding:.55rem 0;border-top:1px solid rgba(255,255,255,.18);font-size:.85rem;}
.ds:first-of-type{border-top:none;} .ds b{font-size:1rem;}

/* streamlit widgets */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px;background:#fff;border-color:var(--line);}
.stButton>button{border-radius:12px;font-weight:700;padding:.6rem 1rem;border:1px solid var(--line);}
.stButton>button[kind="primary"]{background:var(--blue);border:none;color:#fff;}
.stButton>button[kind="primary"]:hover{background:#1A43BF;}
.stTextInput input{border-radius:12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(s: str) -> str:
    """Strip indentation so Markdown does not treat HTML as a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def sparkline(values, w=420, h=110):
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    step = w / (len(values) - 1)
    pts = [(i * step, h - 14 - (v - lo) / span * (h - 32)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} {line} {w},{h}"
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#1D4ED8"/>' for x, y in pts)
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">
      <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#1D4ED8" stop-opacity=".28"/><stop offset="1" stop-color="#1D4ED8" stop-opacity="0"/>
      </linearGradient></defs>
      <polygon points="{area}" fill="url(#g)"/>
      <polyline points="{line}" fill="none" stroke="#1D4ED8" stroke-width="2.5"/>{dots}</svg>"""


# --------------------------------------------------------------------------
# 3. STATE
# --------------------------------------------------------------------------
DEMOS = {
    "Critical (chest + low SpO2)": dict(age=62, gender="Male", comorb="None", hr=112, rr=26,
                                        sbp=88, spo2=89, temp=38.8, glu=210, gcs=13, pain=8),
    "Moderate (fever, stable)": dict(age=45, gender="Female", comorb="Diabetes", hr=98, rr=20,
                                     sbp=118, spo2=95, temp=38.2, glu=160, gcs=15, pain=4),
    "Low risk (routine)": dict(age=29, gender="Female", comorb="None", hr=72, rr=16,
                               sbp=120, spo2=99, temp=36.8, glu=95, gcs=15, pain=1),
}
for k, v in DEMOS["Low risk (routine)"].items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("page", "Dashboard")
st.session_state.setdefault("history", [])
st.session_state.setdefault("latest", None)


def load_demo():
    name = st.session_state.get("demo_pick")
    if name in DEMOS:
        for k, v in DEMOS[name].items():
            st.session_state[k] = v


def go(page):
    st.session_state.page = page


# --------------------------------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(H('<div class="brand"><i>🩺</i>TriageCare</div>'), unsafe_allow_html=True)
    st.radio("Navigation", ["Dashboard", "New assessment", "Latest result", "About"],
             key="page", label_visibility="collapsed")
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.selectbox("Load demo scenario", list(DEMOS), index=None, key="demo_pick",
                 placeholder="Choose scenario", on_change=load_demo)
    st.markdown(H('<p class="side-note">Use demo values only. Do not enter identifiable patient data.</p>'),
                unsafe_allow_html=True)
    st.markdown(H("""<div class="user-card"><div class="avatar">DF</div>
        <div><b>Dr. Foster</b><small>Clinician</small></div></div>"""), unsafe_allow_html=True)

hour = datetime.now().hour
greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
hist = st.session_state.history
n_done = len(hist)
n_high = sum(1 for h in hist if h["label"] == "High")


# --------------------------------------------------------------------------
# 5. PAGES
# --------------------------------------------------------------------------
def topbar(title=None, sub=None):
    c1, c2, c3 = st.columns([3, 2.4, 1.1], vertical_alignment="center")
    with c1:
        st.markdown(H(f"""<div class="hello"><h1>{title or greet + ", Dr. Foster"}</h1>
            <p>{sub or "Here is your triage overview for today."}</p></div>"""), unsafe_allow_html=True)
    with c2:
        st.text_input("Search", placeholder="Search patients, encounters, notes",
                      label_visibility="collapsed")
    with c3:
        st.button("+ New patient", type="primary", use_container_width=True,
                  on_click=go, args=("New assessment",))


def page_dashboard():
    topbar()
    left, right = st.columns([2.3, 1], gap="medium")

    with left:
        a, b = st.columns([1, 1.25], gap="medium")
        pct = min(100, int(n_done / 8 * 100)) if n_done else 0
        with a:
            st.markdown(H(f"""<div class="card"><h3>Today's progress</h3>
                <div class="sub">Assessments completed</div>
                <div class="ring-wrap"><div class="ring" style="--p:{pct}"><b>{pct}%</b></div>
                <div class="pill-row"><div class="stat on"><b>{n_done}</b><span>Done</span></div>
                <div class="stat"><b>{n_high}</b><span>High risk</span></div></div></div></div>"""),
                unsafe_allow_html=True)
        with b:
            st.markdown(H(f"""<div class="card"><h3>Assessment analytics</h3>
                <div class="sub">Assessments per day, this week</div>
                {sparkline([3, 5, 4, 7, 9, 6, 8, max(n_done, 2)])}</div>"""), unsafe_allow_html=True)

        rows = "".join(
            f"""<tr><td>{t}</td><td><span class="pill Wait">{s}</span></td><td><b>{n}</b><br>
            <span style="color:#6B7799;font-size:.75rem">{d}</span></td><td>{m}</td><td>{e}</td></tr>"""
            for t, s, n, d, m, e in [
                ("08:20", "Waiting", "Emily Chen", "32 y, Female", "PT-1053", "Follow-up"),
                ("08:35", "Waiting", "Sarah Mitchell", "30 y, Female", "PT-1043", "New patient"),
                ("09:10", "Scheduled", "Omar Khan", "62 y, Male", "PT-1077", "Chest pain"),
            ])
        st.markdown(H(f"""<div class="card"><h3>Scheduled patients</h3>
            <div class="sub">Demo data. Replace with your patient source.</div>
            <table class="pt"><tr><th>Time</th><th>Status</th><th>Patient</th><th>MRN</th><th>Type</th></tr>
            {rows}</table></div>"""), unsafe_allow_html=True)
        st.button("Start new assessment", type="primary", on_click=go, args=("New assessment",))

    with right:
        st.markdown(H("""<div class="card"><h3>Triage assistant</h3><div class="assist"><div class="orb"></div>
            <b>Ready to help</b><div class="sub">Enter vital signs to get a risk estimate and the factors behind it.</div></div></div>"""),
                    unsafe_allow_html=True)
        last = hist[-1]["label"] if hist else "None yet"
        st.markdown(H(f"""<div class="card dark"><h3>Daily stats</h3><div class="sub">Snapshot of today</div>
            <div class="ds"><span>Assessments</span><b>{n_done}</b></div>
            <div class="ds"><span>High risk</span><b>{n_high}</b></div>
            <div class="ds"><span>Last result</span><b>{last}</b></div></div>"""), unsafe_allow_html=True)


def render_result(res):
    probs, drivers = res["probs"], res["drivers"]
    label = max(probs, key=probs.get)
    color = {"Low": "#16A34A", "Medium": "#D97706", "High": "#DC2626"}[label]
    action = {"Low": "Routine care", "Medium": "Prompt clinical review", "High": "Immediate clinical review"}[label]
    if res["source"] == "demo":
        st.markdown('<div class="banner">Demo scoring is active. Connect your trained model in '
                    '<code>predict_triage()</code> to show real predictions.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1:
        bars = "".join(
            f"""<div class="prob"><span class="n">{k}</span><div class="bar"><i style="width:{v*100:.1f}%;background:{ {'Low':'#16A34A','Medium':'#D97706','High':'#DC2626'}[k] }"></i></div>
            <span class="v">{v*100:.1f}%</span></div>""" for k, v in probs.items())
        st.markdown(H(f"""<div class="result" style="--c:{color}"><div class="lab">Model-estimated category</div>
            <div class="big">{label} risk</div><div>{action}</div></div>
            <div class="card" style="margin-top:1rem"><h3>Risk probability</h3>{bars}</div>"""),
                    unsafe_allow_html=True)
    with c2:
        mx = max((w for _, _, w in drivers), default=1)
        items = "".join(
            f"""<div class="drv"><span class="n">{n}</span><span class="val">{v}</span>
            <div class="bar"><i style="width:{w/mx*100:.0f}%"></i></div></div>""" for n, v, w in drivers
        ) or "<div class='sub'>No abnormal drivers found.</div>"
        st.markdown(H(f"""<div class="card"><h3>Why this result</h3>
            <div class="sub">Factors that pushed the risk up the most</div>{items}</div>
            <div class="card"><h3>Model confidence</h3><div style="font-size:1.6rem;font-weight:800">{probs[label]*100:.1f}%</div>
            <div class="sub">Review with clinical judgment.</div></div>"""), unsafe_allow_html=True)


def page_new():
    topbar("New assessment", "Enter vital signs and review the estimated triage category.")
    with st.container(border=True):
        st.markdown("**Patient assessment**")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("##### Patient details")
            st.number_input("Age (years)", 0, 120, key="age")
            st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
            st.selectbox("Comorbidity", ["None", "Diabetes", "Hypertension", "Heart disease",
                                         "Respiratory disease", "Kidney disease"], key="comorb")
            st.slider("Heart rate (BPM)", 30, 220, key="hr")
            st.slider("Respiratory rate", 5, 50, key="rr")
        with c2:
            st.markdown("##### Vital signs")
            st.slider("Systolic blood pressure (mmHg)", 50, 220, key="sbp")
            st.slider("SpO₂ (%)", 60, 100, key="spo2")
            st.number_input("Temperature (°C)", 30.0, 43.0, step=0.1, key="temp")
            st.number_input("Blood glucose (mg/dL)", 20, 600, key="glu")
            st.selectbox("GCS mental score", list(range(15, 2, -1)), key="gcs")
            st.slider("Pain score (0–10)", 0, 10, key="pain")
        run = st.button("Run triage assessment", type="primary", use_container_width=True)

    if run:
        keys = ["age", "gender", "comorb", "hr", "rr", "sbp", "spo2", "temp", "glu", "gcs", "pain"]
        inputs = {k: st.session_state[k] for k in keys}
        res = predict_triage(inputs)
        res["label"] = max(res["probs"], key=res["probs"].get)
        res["inputs"] = inputs
        st.session_state.latest = res
        st.session_state.history.append(res)

    if st.session_state.latest:
        st.markdown("### Triage assessment")
        render_result(st.session_state.latest)


def page_latest():
    topbar("Latest result", "The most recent assessment in this session.")
    if st.session_state.latest:
        render_result(st.session_state.latest)
    else:
        st.info("No assessment yet. Open New assessment and run one.")
        st.button("Start new assessment", type="primary", on_click=go, args=("New assessment",))


def page_about():
    topbar("About", "How TriageCare works.")
    st.markdown(H("""<div class="card"><h3>Clinical decision support</h3>
        <div class="sub">TriageCare estimates a triage category (Low, Medium, High) from vital signs
        and shows which vitals drove the result using SHAP explanations.
        It supports, and never replaces, clinical judgment. Use demo data only.</div></div>"""),
                unsafe_allow_html=True)


{"Dashboard": page_dashboard, "New assessment": page_new,
 "Latest result": page_latest, "About": page_about}[st.session_state.page]()