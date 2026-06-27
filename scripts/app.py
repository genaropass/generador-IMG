import streamlit as st
import os, sys, shutil, tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_pipeline as sp

st.set_page_config(page_title="DigPatho — Generacion Sintetica", layout="wide")

# ── Estilos clínicos ────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 1.5rem; font-weight: 600; color: #1a2e44; }
    h2, h3 { font-size: 1rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; }
    .stButton > button {
        background: #1a2e44; color: white; border: none;
        border-radius: 4px; padding: 0.5rem 1.5rem;
        font-size: 0.9rem; width: 100%;
    }
    .stButton > button:hover { background: #2c5282; }
    .metric-box {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 4px; padding: 1rem; text-align: center;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #1a2e44; }
    .metric-lbl { font-size: 0.78rem; color: #666; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("DigPatho — Generacion Sintetica de Imagenes Histopatologicas")
st.markdown("Genera variantes sinteticas de imagenes WSI mediante deformacion elastica fractal (fBm + Weierstrass) y augmentacion de tincion (Macenko). El resultado es un dataset listo para entrenamiento.")

st.markdown("---")

col_in, col_out = st.columns([1, 2])

with col_in:
    st.subheader("Parametros de Entrada")

    uploaded = st.file_uploader("Imagen de tejido (JPEG / PNG)", type=["jpg", "jpeg", "png"])
    grade    = st.selectbox("Grado HER2", ["0", "1+", "2+", "3+"], index=3)
    n_var    = st.slider("Variantes sinteticas", min_value=1, max_value=8, value=3)
    alpha    = st.slider("Intensidad de deformacion (alpha)", min_value=10, max_value=80, value=35,
                         help="10 = deformacion suave | 80 = deformacion agresiva")
    draw_pts = st.checkbox("Generar copias de auditoria visual (con puntos verdes superpuestos)", value=False,
                           help="Si se activa, por cada imagen sintetica generada se guardara una copia extra con los puntos detectados dibujados encima, para que el patologo pueda verificarlos visualmente. Estas imagenes terminan con el sufijo '_auditoria.png'.")

    st.markdown(" ")
    run_btn = st.button("Generar Dataset Sintetico")

with col_out:
    st.subheader("Resultado")

    result_placeholder = st.empty()

    if run_btn:
        if uploaded is None:
            st.error("Selecciona una imagen antes de continuar.")
        else:
            work_dir = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico_v2"
            os.makedirs(work_dir, exist_ok=True)

            # Guardar imagen temporal
            tmp_path = os.path.join(work_dir, "input_temp.jpg")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Ejecutando pipeline (A → SAM 2 → Fractal → Deformacion → Macenko)..."):
                try:
                    comp_path = sp.run(tmp_path, work_dir, n_variants=n_var, alpha=alpha, draw_points=draw_pts)
                    st.success("Pipeline completado.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # Imagen comparativa
            if comp_path and os.path.exists(comp_path):
                st.image(Image.open(comp_path), caption="Original vs Variantes Sinteticas", use_column_width=True)

            # Metricas
            ds_dir = os.path.join(work_dir, "dataset_sintetico")
            n_imgs = len([f for f in os.listdir(ds_dir) if f.endswith(".png")]) if os.path.exists(ds_dir) else 0
            n_csv  = len([f for f in os.listdir(ds_dir) if f.endswith(".csv")]) if os.path.exists(ds_dir) else 0

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-box"><div class="metric-val">{n_imgs}</div><div class="metric-lbl">Imagenes generadas</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-val">{n_csv}</div><div class="metric-lbl">Archivos de etiquetas (CSV)</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-val">{alpha}</div><div class="metric-lbl">Alpha de deformacion</div></div>', unsafe_allow_html=True)

            st.markdown(" ")

            # Descargar ZIP
            zip_path = os.path.join(work_dir, "digpatho_sintetico.zip")
            shutil.make_archive(zip_path.replace(".zip", ""), "zip", ds_dir)
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Descargar dataset completo (.zip)",
                    data=f,
                    file_name=f"digpatho_sintetico_grado{grade}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
