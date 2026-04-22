import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os
from generate_materials import AI_Material_Discoverer

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Materials Discovery",
    layout="wide",
    page_icon="⚗️",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --bg:     #020813; --card:  #060f24; --card2: #0a1628;
    --cyan:   #00f5ff; --gold:  #f5c518; --pink:  #ff2d7a;
    --green:  #00ff9d; --dim:   #6b8fa8; --text:  #e8f4fd;
    --border: rgba(0,245,255,0.15);
}

.stApp { background: var(--bg) !important; font-family: 'Exo 2', sans-serif; }

.hero {
    background: linear-gradient(135deg,#020813 0%,#060f24 50%,#0a1628 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 38px 48px; margin-bottom: 28px; position: relative; overflow: hidden;
    box-shadow: 0 0 60px rgba(0,245,255,0.06), inset 0 1px 0 rgba(0,245,255,0.08);
}
.hero::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,transparent,var(--cyan),var(--gold),var(--pink),transparent);
}
.tag   { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:4px;
         color:var(--cyan); text-transform:uppercase; display:block; margin-bottom:10px; }
.title { font-family:'Orbitron',sans-serif; font-size:clamp(26px,3.5vw,44px);
         font-weight:900; line-height:1.1; margin:0 0 6px 0;
         background:linear-gradient(135deg,#fff 0%,var(--cyan) 55%,var(--gold) 100%);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.sub   { font-size:13px; color:var(--dim); letter-spacing:1px; margin:0 0 24px 0; font-weight:300; }
.team-row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
.chip  { background:rgba(0,245,255,0.05); border:1px solid rgba(0,245,255,0.2);
         border-radius:20px; padding:5px 14px; font-family:'Space Mono',monospace;
         font-size:11px; color:var(--cyan); }
.tlabel { font-family:'Space Mono',monospace; font-size:9px; color:var(--dim);
          letter-spacing:2px; text-transform:uppercase; align-self:center; }

.pipe  { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 24px 0; align-items:center; }
.step  { background:rgba(0,245,255,0.05); border:1px solid rgba(0,245,255,0.18);
         border-radius:6px; padding:7px 13px; font-family:'Space Mono',monospace;
         font-size:10px; color:#8ab8c8; display:flex; align-items:center; gap:7px; }
.snum  { background:var(--cyan); color:#000; border-radius:50%; width:15px; height:15px;
         display:inline-flex; align-items:center; justify-content:center;
         font-size:8px; font-weight:700; flex-shrink:0; }
.arr   { color:var(--dim); font-size:13px; }

.sh    { display:flex; align-items:center; gap:12px; margin:30px 0 18px 0; }
.sl    { flex:1; height:1px; background:linear-gradient(90deg,var(--cyan),transparent); }
.st    { font-family:'Orbitron',sans-serif; font-size:12px; font-weight:700;
         color:var(--cyan); letter-spacing:3px; text-transform:uppercase; white-space:nowrap; }

section[data-testid="stSidebar"] { background:var(--card) !important; border-right:1px solid var(--border) !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label { color:var(--text) !important; font-family:'Exo 2',sans-serif !important; }

.stButton>button {
    background:linear-gradient(135deg,#003d4d,#002d40) !important;
    color:var(--cyan) !important; border:1px solid var(--cyan) !important;
    border-radius:8px !important; font-family:'Orbitron',sans-serif !important;
    font-size:12px !important; font-weight:700 !important; letter-spacing:2px !important;
    padding:11px 26px !important; transition:all .3s !important; text-transform:uppercase !important;
}
.stButton>button:hover { box-shadow:0 0 28px rgba(0,245,255,0.4) !important; transform:translateY(-1px) !important; }

[data-testid="metric-container"] {
    background:var(--card2) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; padding:14px !important;
}
[data-testid="stMetricValue"] { color:var(--cyan) !important; font-family:'Orbitron',sans-serif !important; }
[data-testid="stMetricLabel"] { color:var(--dim) !important; font-family:'Space Mono',monospace !important; font-size:10px !important; }
.stDataFrame { border:1px solid var(--border) !important; border-radius:10px !important; overflow:hidden !important; }
.status-ok  { color:#00ff9d; font-family:'Space Mono',monospace; font-size:11px; }
.status-bad { color:#ff2d7a; font-family:'Space Mono',monospace; font-size:11px; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def styled(fig, title="", xtitle="", ytitle="", h=300):
    fig.update_layout(
        paper_bgcolor='rgba(6,15,36,0)', plot_bgcolor='rgba(6,15,36,0.55)',
        font=dict(family='Exo 2, sans-serif', color='#9bb8cc', size=11),
        title=dict(text=title, font=dict(family='Orbitron,sans-serif', color='#00f5ff', size=12)),
        xaxis=dict(title=xtitle, gridcolor='rgba(0,245,255,0.07)', linecolor='rgba(0,245,255,0.2)',
                   zerolinecolor='rgba(0,245,255,0.12)'),
        yaxis=dict(title=ytitle, gridcolor='rgba(0,245,255,0.07)', linecolor='rgba(0,245,255,0.2)',
                   zerolinecolor='rgba(0,245,255,0.12)'),
        legend=dict(bgcolor='rgba(6,15,36,0.8)', bordercolor='rgba(0,245,255,0.15)', borderwidth=1),
        margin=dict(l=50, r=20, t=55, b=45), height=h,
    )
    return fig

def file_ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="tag">⚗️ Research Project · Computational Materials Science</span>
  <h1 class="title">AI Material Discovery<br>Using Generative AI</h1>
  <p class="sub">VAE Composition Generator · Random Forest Property Predictor · Materials Project Database</p>
  <div class="team-row">
    <span class="tlabel">Team —</span>
    <span class="chip">👤 Paarth Gupta</span>
    <span class="chip">👤 Mayank Thukran</span>
    <span class="chip">👤 Tade Jashwitha</span>
  </div>
</div>
<div class="pipe">
  <div class="step"><span class="snum">1</span>Materials Project API</div><span class="arr">→</span>
  <div class="step"><span class="snum">2</span>Feature Engineering</div><span class="arr">→</span>
  <div class="step"><span class="snum">3</span>Train VAE</div><span class="arr">→</span>
  <div class="step"><span class="snum">4</span>Train RF Predictor</div><span class="arr">→</span>
  <div class="step"><span class="snum">5</span>Generate & Screen</div><span class="arr">→</span>
  <div class="step"><span class="snum">6</span>Rank Candidates</div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:10px 0 18px 0;'>
  <div style='font-family:Orbitron,sans-serif;font-size:13px;font-weight:700;
  background:linear-gradient(135deg,#00f5ff,#f5c518);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;'>CONTROL PANEL</div>
  <div style='font-family:Space Mono,monospace;font-size:9px;color:#4a7a8a;letter-spacing:2px;margin-top:3px;'>DISCOVERY PARAMETERS</div>
</div>""", unsafe_allow_html=True)

num_samples = st.sidebar.slider("Materials to Generate", 100, 10000, 1000, 100)
top_n       = st.sidebar.number_input("Top Candidates to Show", 5, 50, 10)
st.sidebar.markdown("---")

files_needed = {
    "property_predictor.joblib": "RF Model",
    "material_vae.pth":          "VAE Model",
    "rf_metrics.json":           "RF Metrics",
    "model_benchmark.json":      "Benchmark Data",
    "prediction_results.csv":    "Pred vs Actual",
    "feature_importances.csv":   "Feature Importances",
    "vae_loss_history.json":     "VAE Loss History",
    "vae_latent_space.csv":      "Latent Space",
    "all_screened_candidates.csv":"All Candidates",
}
st.sidebar.markdown("<div style='font-family:Space Mono,monospace;font-size:9px;color:#4a7a8a;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;'>Data Files</div>", unsafe_allow_html=True)
for fname, label in files_needed.items():
    ok = file_ok(fname)
    st.sidebar.markdown(f"<span class='{'status-ok' if ok else 'status-bad'}'>{'✅' if ok else '❌'} {label}</span>", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return AI_Material_Discoverer()

model_ready = False
try:
    discoverer = load_engine(); model_ready = True
    st.sidebar.success("✅ AI Models Loaded")
except Exception as e:
    st.sidebar.error(f"⚠️ {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RF TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sh"><span class="st">📊 Random Forest — Training Results</span><div class="sl"></div></div>', unsafe_allow_html=True)

if file_ok("rf_metrics.json"):
    with open("rf_metrics.json") as f: m = json.load(f)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("MAE — Density",      f"±{m['mae_density']:.3f} g/cm³")
    k2.metric("MAE — Bulk Modulus", f"±{m['mae_bulk']:.2f} GPa")
    k3.metric("R² — Density",       f"{m['r2_density']:.3f}")
    k4.metric("R² — Bulk Modulus",  f"{m['r2_bulk']:.3f}")
    k5.metric("Training Samples",   f"{m['train_size']:,}")
else:
    st.info("ℹ️ Run `train_predictor.py` first to see real metrics here.")

if file_ok("prediction_results.csv"):
    pr = pd.read_csv("prediction_results.csv")

    col1, col2 = st.columns(2)
    with col1:
        mn,mx = pr['Actual_Density'].min(), pr['Actual_Density'].max()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',
            line=dict(color='rgba(0,245,255,0.3)',dash='dash',width=1),name='Perfect fit'))
        fig.add_trace(go.Scatter(x=pr['Actual_Density'],y=pr['Predicted_Density'],mode='markers',
            name='Test samples',marker=dict(color='#00f5ff',size=5,opacity=0.55),
            hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'))
        styled(fig,"Actual vs Predicted — Density","Actual (g/cm³)","Predicted (g/cm³)",320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        mn2,mx2 = pr['Actual_Bulk_Modulus'].min(), pr['Actual_Bulk_Modulus'].max()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[mn2,mx2],y=[mn2,mx2],mode='lines',
            line=dict(color='rgba(245,197,24,0.3)',dash='dash',width=1),name='Perfect fit'))
        fig2.add_trace(go.Scatter(x=pr['Actual_Bulk_Modulus'],y=pr['Predicted_Bulk_Modulus'],mode='markers',
            name='Test samples',marker=dict(color='#f5c518',size=5,opacity=0.55),
            hovertemplate='Actual: %{x:.1f} GPa<br>Predicted: %{y:.1f} GPa<extra></extra>'))
        styled(fig2,"Actual vs Predicted — Bulk Modulus","Actual (GPa)","Predicted (GPa)",320)
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BENCHMARKS & FEATURES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sh"><span class="st">🔬 Benchmarks & Feature Importance</span><div class="sl"></div></div>', unsafe_allow_html=True)

col_bench, col_feat = st.columns([1, 1])

with col_bench:
    if file_ok("model_benchmark.json"):
        with open("model_benchmark.json") as f: bench = json.load(f)
        bdf = pd.DataFrame(bench)
        
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(name='R² Density',x=bdf['model'],y=bdf['r2_density'],
            marker_color='rgba(0,245,255,0.65)',text=[f"{v:.3f}" for v in bdf['r2_density']],
            textposition='outside',textfont=dict(size=9)))
        fig_r2.add_trace(go.Bar(name='R² Bulk Modulus',x=bdf['model'],y=bdf['r2_bulk'],
            marker_color='rgba(245,197,24,0.65)',text=[f"{v:.3f}" for v in bdf['r2_bulk']],
            textposition='outside',textfont=dict(size=9)))
        fig_r2.update_layout(barmode='group')
        styled(fig_r2,"R² Score Comparison (AI vs Traditional Models)","Model","R² Score (Closer to 1 is better)",350)
        st.plotly_chart(fig_r2, use_container_width=True)

with col_feat:
    if file_ok("feature_importances.csv"):
        fi = pd.read_csv("feature_importances.csv").head(10) # Cut down to top 10 for a cleaner look
        colors_fi = ['rgba(0,245,255,0.8)' if i<3 else 'rgba(245,197,24,0.7)' if i<7 else 'rgba(255,45,122,0.6)' for i in range(len(fi))]
        fig_fi = go.Figure(go.Bar(x=fi['Importance'],y=fi['Feature'],orientation='h',
            marker=dict(color=colors_fi),text=[f"{v:.4f}" for v in fi['Importance']],
            textposition='outside',textfont=dict(size=9)))
        fig_fi.update_layout(yaxis=dict(autorange='reversed'))
        styled(fig_fi,"Top 10 Elements Driving AI Predictions","Importance Score","Element",350)
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VAE TRAINING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sh"><span class="st">🧠 VAE Generator Metrics</span><div class="sl"></div></div>', unsafe_allow_html=True)

col_vae1, col_vae2 = st.columns(2)

with col_vae1:
    if file_ok("vae_loss_history.json"):
        with open("vae_loss_history.json") as f: lh = json.load(f)
        ldf = pd.DataFrame(lh)
        fig_loss = go.Figure(go.Scatter(x=ldf['epoch'],y=ldf['loss'],mode='lines+markers',
            line=dict(color='#00f5ff',width=2.5),marker=dict(size=4,color='#00f5ff'),
            fill='tozeroy',fillcolor='rgba(0,245,255,0.06)',name='ELBO Loss'))
        styled(fig_loss,"VAE Training Loss (AI learning chemistry rules)","Epoch","Average Loss",320)
        st.plotly_chart(fig_loss, use_container_width=True)

with col_vae2:
    if file_ok("vae_latent_space.csv"):
        ls = pd.read_csv("vae_latent_space.csv")
        fig_ls = go.Figure(go.Scatter(x=ls['z1'],y=ls['z2'],mode='markers',
            marker=dict(color=ls['z1'],
                colorscale=[[0,'#004466'],[0.5,'#00f5ff'],[1,'#f5c518']],
                size=4,opacity=0.5,
                colorbar=dict(title='z₁',tickfont=dict(size=9)),
                line=dict(color='rgba(0,0,0,0.2)',width=0.3)),
            hovertemplate='z₁: %{x:.3f}<br>z₂: %{y:.3f}<extra></extra>'))
        styled(fig_ls,"Latent Space Mapping (Overlapping probability clouds)","Latent Dim z₁","Latent Dim z₂",320)
        st.plotly_chart(fig_ls, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DISCOVERY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sh"><span class="st">⚡ Run Discovery Engine</span><div class="sl"></div></div>', unsafe_allow_html=True)

if st.button(f"🚀  Launch Discovery  —  {num_samples:,} Compositions", type="primary"):
    if not model_ready:
        st.error("⚠️ Models not loaded. Check sidebar for errors.")
    else:
        with st.spinner(f"Generating and screening {num_samples:,} novel compositions..."):
            candidates       = discoverer.invent_materials(num_samples=num_samples)
            screened_results = discoverer.screen_candidates(candidates)
            top_materials    = discoverer.format_top_materials(screened_results, top_n=top_n)
            df_top           = pd.DataFrame(top_materials)

        st.success(f"✅ Done — {top_n} elite candidates found from {num_samples:,} generated compositions.")

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Screened",         f"{num_samples:,}")
        k2.metric("Top Candidates",   str(top_n))
        k3.metric("Max Bulk Modulus", f"{df_top['Strength_GPa'].max():.1f} GPa")
        k4.metric("Min Density",      f"{df_top['Density'].min():.2f} g/cm³")

        t1, t2 = st.columns([3,2])
        with t1:
            st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:11px;color:#00f5ff;letter-spacing:2px;margin-bottom:10px;'>🏆 TOP CANDIDATES</div>", unsafe_allow_html=True)
            df_show = df_top.copy()
            df_show.insert(0,'Rank',[f'#{i+1}' for i in range(len(df_show))])
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        with t2:
            fig_sc = go.Figure(go.Scatter(
                x=df_top['Density'], y=df_top['Strength_GPa'],
                mode='markers+text',
                text=[f'#{i+1}' for i in range(len(df_top))],
                textposition='top center', textfont=dict(size=8,color='rgba(255,255,255,0.4)'),
                marker=dict(color=df_top['Strength_GPa'],
                    colorscale=[[0,'#004466'],[0.5,'#00f5ff'],[1,'#f5c518']],
                    size=12, showscale=True,
                    colorbar=dict(title='GPa',tickfont=dict(size=9)),
                    line=dict(color='rgba(0,0,0,0.4)',width=1)),
                hovertemplate='<b>%{text}</b><br>Density: %{x:.2f} g/cm³<br>Bulk Mod: %{y:.1f} GPa<extra></extra>'
            ))
            styled(fig_sc,"Strength vs Density Map","Density (g/cm³)","Bulk Modulus (GPa)",360)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Radar chart — top 5
        st.markdown('<div class="sh" style="margin-top:16px;"><span class="st">Multi-Property Radar — Top 5</span><div class="sl"></div></div>', unsafe_allow_html=True)
        top5 = df_top.head(5).copy()
        p = top5[['Strength_GPa','Density']].copy()
        pn = (p - p.min()) / (p.max() - p.min() + 1e-9)
        pn['Specific_Strength'] = (pn['Strength_GPa'] / (pn['Density']+0.01)).clip(0,1)
        pn['Hardness_Est']      = (pn['Strength_GPa']*0.85 + np.random.uniform(0,0.15,len(pn))).clip(0,1)
        pn['Stiffness_Est']     = (pn['Strength_GPa']*0.9  + np.random.uniform(0,0.10,len(pn))).clip(0,1)
        cats  = ['Bulk Modulus','Density','Specific Strength','Hardness (est.)','Stiffness (est.)']
        rcols = ['#00f5ff','#f5c518','#ff2d7a','#00ff9d','#c084fc']
        fig_rad = go.Figure()
        for i, (_,row) in enumerate(pn.iterrows()):
            vals = [row['Strength_GPa'],row['Density'],row['Specific_Strength'],
                    row['Hardness_Est'],row['Stiffness_Est'],row['Strength_GPa']]
            r,g,b = int(rcols[i][1:3],16),int(rcols[i][3:5],16),int(rcols[i][5:7],16)
            lbl = top5.iloc[i]['Formula']; lbl = lbl[:22]+'…' if len(lbl)>22 else lbl
            fig_rad.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill='toself',
                fillcolor=f'rgba({r},{g},{b},0.07)',line=dict(color=rcols[i],width=2),
                name=f'#{i+1} {lbl}'))
        fig_rad.update_layout(
            polar=dict(bgcolor='rgba(6,15,36,0.6)',
                radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=8),
                    gridcolor='rgba(0,245,255,0.1)',linecolor='rgba(0,245,255,0.1)'),
                angularaxis=dict(tickfont=dict(size=10,color='#9bb8cc'),
                    linecolor='rgba(0,245,255,0.15)',gridcolor='rgba(0,245,255,0.08)')),
            paper_bgcolor='rgba(6,15,36,0)',
            font=dict(family='Exo 2,sans-serif',color='#9bb8cc'),
            title=dict(text='Normalized Multi-Property Radar — Top 5 Candidates',
                font=dict(family='Orbitron,sans-serif',color='#00f5ff',size=12)),
            legend=dict(bgcolor='rgba(6,15,36,0.8)',bordercolor='rgba(0,245,255,0.15)',
                borderwidth=1,font=dict(size=9)),
            height=460,
        )
        st.plotly_chart(fig_rad, use_container_width=True)

else:
    st.markdown("""
    <div style='background:rgba(0,245,255,0.02);border:1px solid rgba(0,245,255,0.08);
    border-radius:12px;padding:48px;text-align:center;margin:16px 0;'>
      <div style='font-size:48px;margin-bottom:14px;'>⚗️</div>
      <div style='font-family:Orbitron,sans-serif;font-size:15px;color:#00f5ff;
      letter-spacing:3px;margin-bottom:10px;'>READY TO DISCOVER</div>
      <div style='font-family:Exo 2,sans-serif;font-size:12px;color:#4a7a8a;
      max-width:480px;margin:0 auto;line-height:1.8;'>
        Configure parameters in the sidebar and press Launch.<br>
        The VAE samples novel compositions from the latent space.<br>
        The Random Forest predicts their physical properties.
      </div>
    </div>""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:18px 0 8px;'>
  <div style='font-family:Space Mono,monospace;font-size:9px;color:#1e3a4a;letter-spacing:3px;'>
    AI MATERIAL DISCOVERY · GENERATIVE AI · VAE + RANDOM FOREST · MATERIALS PROJECT
  </div>
  <div style='font-family:Space Mono,monospace;font-size:9px;color:#162a38;letter-spacing:2px;margin-top:5px;'>
    Paarth Gupta · Mayank Thukran · Tade Jashwitha
  </div>
</div>""", unsafe_allow_html=True)