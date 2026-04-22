import streamlit as st
import pandas as pd
# We import your exact backend engine!
from generate_materials import AI_Material_Discoverer

# 1. Setup the Web Page
st.set_page_config(page_title="AI Materials Discovery", layout="wide", page_icon="🧪")

st.title("🧪 AI-Driven Materials Discovery System")
st.markdown("""
This system uses a **Variational Autoencoder (VAE)** to hallucinate novel chemical compositions 
and a **Random Forest Regressor** to predict their physical properties.
""")

# 2. Load the AI Models (Cached so it doesn't reload on every button click)
@st.cache_resource
def load_ai_engine():
    return AI_Material_Discoverer()

try:
    discoverer = load_ai_engine()
    st.sidebar.success("✅ AI Models Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

# 3. User Interface Controls
st.sidebar.header("Discovery Parameters")
num_samples = st.sidebar.slider("Number of Materials to Generate", min_value=100, max_value=10000, value=1000, step=100)
top_n = st.sidebar.number_input("How many top candidates to show?", min_value=5, max_value=50, value=10)

# 4. The Main Execution Button
if st.button("🚀 Discover New Materials", type="primary"):
    with st.spinner(f"AI is generating and screening {num_samples} novel compositions..."):
        
        # Run the backend pipeline
        candidates = discoverer.invent_materials(num_samples=num_samples)
        screened_results = discoverer.screen_candidates(candidates)
        top_materials = discoverer.format_top_materials(screened_results, top_n=top_n)
        
        # Convert results to a clean table
        df_display = pd.DataFrame(top_materials)
        
        st.success("Discovery Complete!")
        
        # Create two columns for the dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Top AI-Generated Candidates")
            st.dataframe(df_display, use_container_width=True)
            
        with col2:
            st.subheader("📊 Property Distribution")
            # Create a simple scatter plot of the results
            st.scatter_chart(data=df_display, x="Density", y="Strength_GPa", color="#ff4b4b")