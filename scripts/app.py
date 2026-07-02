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
            # Ruta corregida a la carpeta 'generador'
            work_dir = r"E:\Genaro\Desktop\Digital pathologies\generador\output_sintetico_v2"
            os.makedirs(work_dir, exist_ok=True)

            # Guardar imagen temporal
            tmp_path = os.path.join(work_dir, "input_temp.jpg")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Ejecutando pipeline (A -> SAM 2 -> Fractal -> Deformacion -> Macenko)..."):
                try:
                    comp_path = sp.run(tmp_path, work_dir, n_variants=n_var, alpha=alpha, grade=grade, draw_points=draw_pts)
                    st.success("Pipeline completado.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # Imagen comparativa
            if comp_path and os.path.exists(comp_path):
                st.image(Image.open(comp_path), caption="Original vs Variantes Sinteticas", use_column_width=True)

            # Metricas Generales
            ds_dir = os.path.join(work_dir, "dataset_sintetico")
            n_imgs = len([f for f in os.listdir(ds_dir) if f.endswith(".png") and not f.endswith("_auditoria.png")]) if os.path.exists(ds_dir) else 0
            n_csv  = len([f for f in os.listdir(ds_dir) if f.endswith(".csv")]) if os.path.exists(ds_dir) else 0

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-box"><div class="metric-val">{n_imgs}</div><div class="metric-lbl">Imagenes generadas</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-val">{n_csv}</div><div class="metric-lbl">Archivos de etiquetas (CSV)</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-val">{alpha}</div><div class="metric-lbl">Alpha de deformacion</div></div>', unsafe_allow_html=True)

            # Mostrar nuevas metricas de Minkowski & Zetas
            csv_files = sorted([f for f in os.listdir(ds_dir) if f.endswith("_labels.csv")]) if os.path.exists(ds_dir) else []
            if csv_files:
                import pandas as pd
                df_labels = pd.read_csv(os.path.join(ds_dir, csv_files[0]))
                
                st.markdown("---")
                st.subheader("Estadisticas Geometricas y Fractales (Variante 1)")
                
                c1, c2, c3, c4 = st.columns(4)
                mean_mink_dim = df_labels["minkowski_dim"].mean()
                mean_mink_lac = df_labels["minkowski_lacunarity"].mean()
                mean_omega = df_labels["complex_omega"].mean()
                mean_amp = df_labels["complex_amplitude"].mean()
                
                c1.metric("Dimens. Minkowski (D)", f"{mean_mink_dim:.4f}")
                c2.metric("Lacun. Minkowski", f"{mean_mink_lac:.4f}")
                c3.metric("Frec. Compleja (omega)", f"{mean_omega:.4f}")
                c4.metric("Amplitud Compleja", f"{mean_amp:.4f}")
                
                st.markdown("**Muestra de las primeras 5 celulas remapeadas:**")
                st.dataframe(df_labels.head(5).style.format({
                    "x": "{:.1f}", "y": "{:.1f}", "df_original": "{:.3f}", "lacunarity": "{:.3f}",
                    "minkowski_dim": "{:.3f}", "upper_minkowski": "{:.3f}", "lower_minkowski": "{:.3f}",
                    "minkowski_lacunarity": "{:.3f}", "complex_omega": "{:.3f}", "complex_amplitude": "{:.3f}"
                }))

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
