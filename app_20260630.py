#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import base64
import urllib.parse as _url
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher

# Try to use rapidfuzz if available
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="CDH Synapse | KPJ Academic",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_BASENAME  = "subspecialityprediction"
FUZZY_THRESHOLD = 65

# --- SYNC THEME (Warm & Premium) ---
COLOR_BG      = "#F3F0EB"   
COLOR_PRIMARY = "#8C7C68"   
COLOR_TITLE   = "#6D6256"   
COLOR_HOVER   = "#595046"   
COLOR_TEXT    = "#5D5348"   

# ================= LOGIC: FELLOWSHIP RULES =================
FELLOWSHIP_RULES = {
    "Interventional Cardiology": [
        "pci", "angioplasty", "stent", "cto", "chronic total occlusion", 
        "coronary", "angiogram", "radial", "scaffold", "thrombectomy", 
        "intervention", "endovascular", "peripheral vascular", "complex pci"
    ],
    "Non-Invasive & Geriatric Cardiology": [
        "non invasive", "stress test", "treadmill", "holter", "ecg", 
        "diagnostic", "monitoring", "ambulatory", "geriatric", "elderly", "general cardiology"
    ],
    "Cardiac Electrophysiology": [
        "electrophysiology", "pacemaker", "icd", "crt", "ablation", 
        "arrhythmia", "atrial fibrillation", "device therapy", "pacing", 
        "ep study", "device follow-up", "remote monitoring"
    ],
    "Heart Failure & Critical Care": [
        "heart failure", "cardiomyopathy", "lvad", "transplant", 
        "ejection fraction", "pump failure", "amyloidosis", "mechanical support", 
        "mcs", "critical care", "intensive care", "cic", "shock"
    ],
    "Structural Heart & Valvular": [
        "tavi", "mitraclip", "valvular", "aortic stenosis", 
        "mitral regurgitation", "structural", "valve", "tricuspid", "laa closure"
    ],
    "Cardiac Imaging": [
        "echocardiography", "ctca", "cmr", "mri", "strain imaging", 
        "nuclear", "calcium score", "pet scan", "cardiac ct", "multimodality"
    ],
    "Cardiovascular Genetics": [
        "genetics", "inherited", "dna", "familial", "genome", 
        "hypertrophic cardiomyopathy", "hcm", "channelopathy", 
        "brugada", "long qt", "genetic counseling"
    ],
    "Preventive & Metabolic Medicine": [
        "lipid", "cholesterol", "statin", "preventive", "risk factor", 
        "hypertension", "diabetes", "metabolic", "obesity", "smoking", 
        "cessation", "cardiovascular risk", "adherence", "secondary prevention",
        "cardio-renal", "metabolic syndrome", "post-mi"
    ],
    "Pulmonary Hypertension": [
        "pulmonary hypertension", "right heart", "pah", "embolism", "ph"
    ],
    "Sports Cardiology": [
        "athlete", "sports", "endurance", "exercise physiology", "screening", "sudden death"
    ],
    "Cardio-Oncology": [
        "cardiotoxicity", "cancer", "chemotherapy", "anthracycline", 
        "oncology", "radiation", "immunotherapy"
    ],
    "Cardio-Obstetrics": [
        "pregnancy", "maternal", "foetal", "pregnant", "peripartum"
    ],
    "Paediatric & ACHD": [
        "paediatric", "congenital", "child", "infant", "tetralogy", 
        "asd", "vsd", "blue baby", "achd", "adult congenital"
    ]
}

# ================= LOGIC FUNCTIONS =================
def get_hospital_saturation(df):
    counts = {k: 0 for k in FELLOWSHIP_RULES.keys()}
    existing_subs = df["Subspecialty/Fellowship"].dropna().astype(str).tolist()
    
    for existing in existing_subs:
        existing_lower = existing.lower()
        matched = False
        # 1. Match Category Name
        for category in FELLOWSHIP_RULES.keys():
            if category.split(" & ")[0].lower() in existing_lower:
                counts[category] += 1
                matched = True
                break
        # 2. Match Keywords
        if not matched:
            for category, keywords in FELLOWSHIP_RULES.items():
                for kw in keywords:
                    if kw in existing_lower:
                        counts[category] += 1
                        matched = True
                        break
                if matched: break
    return counts

def predict_gap_filling_fellowship(df, interest, pub_research, ong_research, doctor_region):
    # --- NEW LOGIC: Filter Dataset by Region First ---
    # We only care about saturation in the doctor's specific region
    if doctor_region:
        regional_df = df[df["Region"] == doctor_region]
        region_text = f"in {doctor_region}"
    else:
        regional_df = df # Fallback if region is unknown
        region_text = "nationally"

    full_text = f"{str(interest)} {str(pub_research)} {str(ong_research)}".lower()
    candidates = []
    
    for fellowship, keywords in FELLOWSHIP_RULES.items():
        match_count = 0
        matched_keywords = []
        for kw in keywords:
            if kw in full_text:
                match_count += 1
                matched_keywords.append(kw)
        if match_count > 0:
            conf = min(50 + (match_count * 15), 98)
            candidates.append({"fellowship": fellowship, "score": match_count, "confidence": conf, "keywords": matched_keywords})
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    if not candidates: return None, 0, "No matching keywords found in profile."

    # Calculate saturation using ONLY the regional data
    saturation_map = get_hospital_saturation(regional_df)
    selected = None
    skipped_log = []
    
    for cand in candidates:
        f_name = cand['fellowship']
        existing_count = saturation_map.get(f_name, 0)
        
        # Check if gap exists IN THIS REGION
        if existing_count == 0:
            selected = cand
            selected['status'] = "Strategic Gap"
            break
        else:
            # Log that we skipped it because the REGION already has it
            skipped_log.append(f"Skipped <b>{f_name}</b> ({cand['confidence']}% match) because {existing_count} specialists already exist {region_text}.")
    
    if selected is None:
        selected = candidates[0]
        selected['status'] = "Competitive"

    reason_str = f"<b>Selected Strategy: {selected['status']} ({region_text})</b><br>Derived from: {', '.join(selected['keywords'])}."
    if skipped_log:
        reason_str += "<br><br><div style='background:#EFEBE6; padding:8px; border-radius:5px; border-left:3px solid #6D6256; font-size:0.8em; color:#5D5348;'>" + "<b>⚠️ Strategic Bypass:</b><br>" + "<br>".join(skipped_log) + "</div>"
        
    return selected['fellowship'], selected['confidence'], reason_str

# ================= UI: STYLING =================
def inject_custom_css():
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {COLOR_BG}; font-family: 'Open Sans', 'Segoe UI', sans-serif; color: {COLOR_TEXT}; }}
        [data-testid="stSidebar"] {{ background-color: #EAE6E1; border-right: 1px solid #DCD5CD; }}
        
        .kpj-header-bar {{ background-color: {COLOR_BG}; padding: 15px 25px; border-bottom: 2px solid {COLOR_PRIMARY}; margin-bottom: 25px; display: flex; align-items: center; gap: 20px; }}
        .kpj-logo {{ height: 60px; width: auto; }}
        .kpj-header-text-block {{ border-left: 2px solid {COLOR_PRIMARY}; padding-left: 20px; }}
        .kpj-header-title {{ color: {COLOR_TITLE}; font-size: 28px; font-weight: 800; margin: 0; line-height: 1.1; letter-spacing: -0.5px; }}
        .kpj-header-subtitle {{ color: #8C7C68; font-size: 14px; font-weight: 600; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}
        
        .stTextInput input {{ background-color: #FFFFFF !important; color: #5D5348 !important; border: 1px solid #D0C9C0 !important; border-radius: 8px !important; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }}
        div[data-testid="stForm"] {{ border: none; padding: 0; }}
        .table-container {{ background: white; border-radius: 12px; box-shadow: 0 4px 15px rgba(93, 83, 72, 0.08); border: 1px solid #EAE6E1; overflow: hidden; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        thead {{ background-color: #EAE6E1; border-bottom: 2px solid #D0C9C0; }}
        th {{ color: {COLOR_TITLE}; font-weight: 700; padding: 15px 20px; text-align: left; text-transform: uppercase; font-size: 12px; }}
        td {{ border-bottom: 1px solid #F3F0EB; padding: 15px 20px; color: {COLOR_TEXT}; vertical-align: middle; }}
        
        .metric-card {{ background: white; padding: 20px; border-radius: 12px; border-left: 6px solid {COLOR_PRIMARY}; box-shadow: 0 4px 15px rgba(93, 83, 72, 0.05); margin-bottom: 10px; }}
        .metric-val {{ font-size: 2.2rem; font-weight: 800; color: {COLOR_TITLE}; }}
        .metric-label {{ font-size: 0.85rem; color: #8C7C68; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }}
        
        div.stButton > button[kind="primary"] {{ background-color: {COLOR_PRIMARY} !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; box-shadow: 0 4px 10px rgba(140, 124, 104, 0.3); }}
        div.stButton > button[kind="primary"]:hover {{ background-color: {COLOR_HOVER} !important; transform: translateY(-1px); }}
        
        a.doctor-link {{ color: {COLOR_PRIMARY}; font-weight: 700; text-decoration: none; }}
        
        .profile-wrapper {{ background: white; border-radius: 16px; box-shadow: 0 10px 30px rgba(93, 83, 72, 0.1); border: 1px solid #EAE6E1; overflow: hidden; margin-top: 10px; }}
        .profile-top-section {{ background: #FBF9F7; padding: 30px; border-bottom: 1px solid #EAE6E1; display: flex; align-items: center; gap: 25px; }}
        .profile-photo {{ width: 100px; height: 100px; border-radius: 50%; border: 4px solid #EAE6E1; object-fit: cover; }}
        .profile-name-block h2 {{ margin: 0; font-size: 1.6rem; color: {COLOR_TITLE}; font-weight: 800; }}
        .profile-details-section {{ padding: 30px; }}
        
        .profile-row-inline {{ margin-bottom: 15px; display: flex; align-items: baseline; border-bottom: 1px dashed #EAE6E1; padding-bottom: 8px; }}
        .profile-label-text {{ font-weight: 700; color: #8C7C68; text-transform: uppercase; font-size: 0.8rem; min-width: 160px; margin-right: 10px; letter-spacing: 0.5px; }}
        .profile-value-text {{ color: {COLOR_TEXT}; font-size: 1rem; line-height: 1.5; flex: 1; font-weight: 500; }}
        
        .ai-box {{ background: #FBF9F7; border: 1px solid #D0C9C0; border-left: 6px solid {COLOR_PRIMARY}; border-radius: 12px; padding: 25px; margin-top: 20px; animation: fadeIn 0.6s; }}
        .ai-title {{ color: {COLOR_TITLE}; font-weight: 800; font-size: 1.2rem; margin-bottom: 10px; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    </style>
    """, unsafe_allow_html=True)

# ================= DATA HELPERS =================
@st.cache_data
def load_profile_image_b64() -> str:
    try:
        with open("person.png", "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError: return ""

@st.cache_data
def load_logo_b64() -> str:
    try:
        with open("logo.png", "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError: return ""

PROFILE_IMG_B64 = load_profile_image_b64()
LOGO_IMG_B64 = load_logo_b64()

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda x: str(x).replace("\xa0", " ").strip())

@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_excel(f"{DATA_BASENAME}.xlsx")
    except Exception:
        try:
            df = pd.read_csv(f"{DATA_BASENAME}.csv", encoding="latin1")
        except:
            st.error(f"Data file '{DATA_BASENAME}.xlsx' or .csv not found.")
            return pd.DataFrame()

    df = clean_headers(df)
    if "Region" not in df.columns: df["Region"] = "Unknown"
    if "Hospital" not in df.columns: df["Hospital"] = "KPJ Specialist Hospital"
    if "DoctorID" not in df.columns:
        df.insert(0, "DoctorID", [f"D{idx+1:03d}" for idx in range(len(df))])
    df["DoctorID"] = df["DoctorID"].astype(str)
    return df

# ================= DASHBOARD PAGE =================
def render_dashboard(df):
    st.markdown(f"<h3 style='color:{COLOR_TITLE}'>📊 Executive Overview</h3>", unsafe_allow_html=True)
    
    total_docs = len(df)
    sub_assigned = df[df["Subspecialty/Fellowship"].notna() & (df["Subspecialty/Fellowship"].astype(str).str.strip() != "")].shape[0]
    pending = total_docs - sub_assigned
    coverage_ratio = int((sub_assigned / total_docs) * 100) if total_docs > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-val">{total_docs}</div><div class="metric-label">Total Specialists</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-val">{sub_assigned}</div><div class="metric-label">Sub-Specialized</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-val">{pending}</div><div class="metric-label">Pending Pathway</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-val">{coverage_ratio}%</div><div class="metric-label">Coverage Ratio</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown(f"<h4 style='color:{COLOR_TITLE}'>🏥 Current Subspecialty Saturation (The \"Existing Strength\")</h4>", unsafe_allow_html=True)
    saturation_data = get_hospital_saturation(df)
    sat_df = pd.DataFrame(list(saturation_data.items()), columns=['Subspecialty', 'Count'])
    
    sat_df_active = sat_df[sat_df['Count'] > 0].copy()
    
    fig_tree = px.treemap(sat_df_active, path=['Subspecialty'], values='Count',
                          color='Count', color_continuous_scale=[COLOR_BG, COLOR_PRIMARY])
    
    fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_tree.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}')
    st.plotly_chart(fig_tree, use_container_width=True)

    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown(f"<h4 style='color:{COLOR_TITLE}'>⚠️ Strategic Gaps (The \"Missing Expert\")</h4>", unsafe_allow_html=True)
        gaps = sat_df[sat_df['Count'] == 0]['Subspecialty'].tolist()
        if gaps:
            for gap in gaps:
                st.markdown(f"❌ **{gap}**")
            st.caption("These areas have 0 specialists. AI will prioritize these for candidates with matching interests.")
        else:
            st.success("Full Spectrum Coverage Achieved!")

    with c_right:
        st.markdown(f"<h4 style='color:{COLOR_TITLE}'>🫀 Pipeline Interests (The \"Future Talent\")</h4>", unsafe_allow_html=True)
        pending_df = df[df["Subspecialty/Fellowship"].isna() | (df["Subspecialty/Fellowship"] == "")]
        if not pending_df.empty:
            interest_counts = {}
            for _, row in pending_df.iterrows():
                text = str(row.get('SpecialInterest', '')).lower()
                for cat, kws in FELLOWSHIP_RULES.items():
                    for kw in kws:
                        if kw in text:
                            interest_counts[cat] = interest_counts.get(cat, 0) + 1
                            break
            
            int_df = pd.DataFrame(list(interest_counts.items()), columns=['Potential Pathway', 'Candidates'])
            int_df = int_df.sort_values('Candidates', ascending=False)
            
            fig_pipe = px.pie(int_df, values='Candidates', names='Potential Pathway', hole=0.4,
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pipe.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_pipe.update_traces(hovertemplate='<b>%{label}</b><br>Candidates: %{value}')
            st.plotly_chart(fig_pipe, use_container_width=True)
        else:
            st.info("No doctors currently pending fellowship.")

# ================= SEARCH PAGE =================
def render_search(df):
    if "q" not in st.session_state: st.session_state["q"] = st.query_params.get("q", "")

    with st.container():
        col_form, col_reset = st.columns([5, 1])
        with col_form:
            with st.form(key="search_form", clear_on_submit=False):
                c_in, c_btn = st.columns([4, 1])
                with c_in: q_input = st.text_input("Search", value=st.session_state["q"], placeholder="Search by keyword, name, or interest...", label_visibility="collapsed")
                with c_btn: is_search = st.form_submit_button("Search", type="primary", use_container_width=True)
        with col_reset:
            if st.button("Reset", type="secondary", use_container_width=True):
                st.session_state["q"] = ""
                if "doctor" in st.query_params: del st.query_params["doctor"]
                st.query_params.clear()
                st.rerun()

    if is_search:
        st.session_state["q"] = q_input
        st.query_params["q"] = q_input
        if "doctor" in st.query_params: del st.query_params["doctor"]
        st.rerun()

    q = st.session_state["q"]
    
    if q.strip():
        df_norm = df.copy()
        search_cols = ["Specialist", "PublishedResearch", "OngoingResearch", "SpecialInterest"]
        df_norm["_search_text"] = df_norm[search_cols].fillna("").agg(" ".join, axis=1).astype(str).str.lower()
        q_norm = q.lower().strip()
        encoded_q = _url.quote_plus(q) if q else ""
        filtered = df[df_norm["_search_text"].str.contains(q_norm, na=False)]
        if filtered.empty:
            def get_fuzzy_score(row_text):
                if _HAS_RAPIDFUZZ: return fuzz.partial_ratio(q_norm, row_text)
                else: return SequenceMatcher(None, q_norm, row_text).ratio() * 100
            df_norm["_score"] = df_norm["_search_text"].apply(get_fuzzy_score)
            filtered = df.loc[df_norm["_score"] >= FUZZY_THRESHOLD]
        st.markdown(f"<div style='color:{COLOR_TITLE}; margin-bottom:10px; font-size:0.9rem;'>Found {len(filtered)} specialists matching your criteria.</div>", unsafe_allow_html=True)
    else:
        filtered = pd.DataFrame()
        st.info("Please enter search terms above to locate specialists.")

    selected_doc_id = st.query_params.get("doctor", None)
    if selected_doc_id:
        col_list, col_profile = st.columns([6, 4], gap="large")
    else:
        col_list, col_profile = st.columns([1, 0.01])

    with col_list:
        if not filtered.empty:
            html_rows = ""
            encoded_q = _url.quote_plus(q) if q else ""
            for _, row in filtered[["DoctorID", "Specialist", "Subspecialty/Fellowship", "PublishedResearch"]].iterrows():
                subsp = row.get("Subspecialty/Fellowship", "")
                is_missing = pd.isna(subsp) or str(subsp).strip() == ""
                bg_style = "background-color: #FFFDF5;" if is_missing else "" 
                status_badge = f"<span style='color:#B08D55; font-size:0.85em; font-weight:600;'>Needs Pathway</span>" if is_missing else str(subsp)
                doc_id = row['DoctorID']
                link = f"?q={encoded_q}&doctor={doc_id}"
                res_text = str(row['PublishedResearch'])
                if len(res_text) > 70: res_text = res_text[:70] + "..."
                if res_text == "nan": res_text = "-"
                html_rows += f'<tr style="{bg_style}"><td><a href="{link}" target="_self" class="doctor-link">{row["Specialist"]}</a></td><td>{status_badge}</td><td style="color:#666;">{res_text}</td></tr>'
            st.markdown(f'<div class="table-container"><table><thead><tr><th width="30%">Specialist Name</th><th width="25%">Subspecialty</th><th width="45%">Research</th></tr></thead><tbody>{html_rows}</tbody></table></div>', unsafe_allow_html=True)

    with col_profile:
        if selected_doc_id:
            doc_data = df[df["DoctorID"] == selected_doc_id]
            if not doc_data.empty:
                record = doc_data.iloc[0].to_dict()
                img_tag = f'<img src="data:image/png;base64,{PROFILE_IMG_B64}" class="profile-photo">' if PROFILE_IMG_B64 else ""
                
                card_html = f"""<div class="profile-wrapper">
<div class="profile-top-section">
{img_tag}
<div class="profile-name-block">
<h2>{record.get('Specialist')}</h2>
<p>🏥 {record.get('Hospital', 'KPJ Specialist Hospital')}</p>
<p>📍 {record.get('Region', 'Malaysia')}</p>
</div>
</div>
<div class="profile-details-section">"""

                fields = [("Specialty", record.get("Specialty")), ("Subspecialty", record.get("Subspecialty/Fellowship")), ("Clinical Interest", record.get("SpecialInterest")), ("Ongoing Research", record.get("OngoingResearch")), ("Published Research", record.get("PublishedResearch"))]
                
                for label, val in fields:
                    if pd.notna(val) and str(val).strip() != "":
                        card_html += f"""<div class="profile-row-inline"><span class="profile-label-text">{label}:</span><span class="profile-value-text">{val}</span></div>"""
                
                card_html += "</div></div>"
                st.markdown(card_html, unsafe_allow_html=True)

                current_sub = record.get("Subspecialty/Fellowship", "")
                if pd.isna(current_sub) or str(current_sub).strip() == "":
                    st.write("") 
                    if st.button("Generate Fellowship Recommendation", type="primary", use_container_width=True):
                        with st.spinner("Analysing clinical gaps and profile matches..."):
                            import time
                            time.sleep(0.5)
                            # --- HERE: WE PASS THE REGION TO THE AI FUNCTION ---
                            sugg, conf, reasoning = predict_gap_filling_fellowship(
                                df, 
                                record.get("SpecialInterest", ""), 
                                record.get("PublishedResearch", ""), 
                                record.get("OngoingResearch", ""),
                                record.get("Region") # Pass Region here
                            )
                        if sugg:
                            st.markdown(f"""<div class="ai-box"><div class="ai-title">Recommended Pathway</div><div style="font-size:1.2rem; font-weight:600; color:#5D5348; margin-bottom:10px;">{sugg}</div><div style="font-size:0.9em; color:#5D5348; line-height:1.4;">{reasoning}</div><div style="margin-top:15px; font-size:0.8em; font-weight:bold; color:#8C7C68;">Confidence Score: {conf}%</div></div>""", unsafe_allow_html=True)
                        else:
                            st.warning(reasoning if reasoning else "Insufficient data.")

# ================= APP CONTROLLER =================
def show():
    inject_custom_css()
    
    with st.sidebar:
        st.markdown(f"<h3 style='color:{COLOR_TITLE}'>Navigation</h3>", unsafe_allow_html=True)
        page = st.radio("Go to", ["Specialist Search", "Executive Dashboard"], label_visibility="collapsed")
        st.markdown("---")

    # Load Data Globally
    df = load_data()
    if df.empty: return

    # Logic for Sidebar Filter (Executive Dashboard only)
    df_filtered = df.copy()
    if page == "Executive Dashboard":
        with st.sidebar:
            st.markdown(f"<div style='color:{COLOR_TITLE}; font-weight:bold; margin-bottom:5px;'>📍 Filter Scope</div>", unsafe_allow_html=True)
            
            all_regions = sorted(df["Region"].dropna().unique().tolist())
            selected_region = st.selectbox("Region", ["All Regions"] + all_regions)
            
            if selected_region != "All Regions":
                avail_hospitals = sorted(df[df["Region"] == selected_region]["Hospital"].dropna().unique().tolist())
                df_filtered = df[df["Region"] == selected_region]
            else:
                avail_hospitals = sorted(df["Hospital"].dropna().unique().tolist())
                
            selected_hospital = st.selectbox("Hospital", ["All Hospitals"] + avail_hospitals)
            
            if selected_hospital != "All Hospitals":
                df_filtered = df_filtered[df_filtered["Hospital"] == selected_hospital]

            st.markdown("---")

    # Sidebar Footer
    with st.sidebar:
        st.markdown("<div style='font-size:0.8em; color:#8C7C68;'>CDH Synapse v2.2<br>Internal Usage Only</div>", unsafe_allow_html=True)

    logo_header_html = f'<img src="data:image/png;base64,{LOGO_IMG_B64}" class="kpj-logo">' if LOGO_IMG_B64 else ""
    st.markdown(f"""<div class="kpj-header-bar">{logo_header_html}<div class="kpj-header-text-block"><div class="kpj-header-title">CDH Synapse</div><div class="kpj-header-subtitle">Specialist Career & Research Pathway</div></div></div>""", unsafe_allow_html=True)

    if page == "Specialist Search": 
        render_search(df)
    else: 
        render_dashboard(df_filtered)

if __name__ == "__main__":
    show()


# In[ ]:




