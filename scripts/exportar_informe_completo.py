"""
DigPatho – Exportador del Informe Unificado (Etapas 1-9)
Genera un único HTML con todo el pipeline documentado.
"""
import base64, os, json

def b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

BASE = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico"
DS   = os.path.join(BASE, "dataset_sintetico")

img_e12  = b64(os.path.join(BASE, "blueprint_etapa1_2.png"))
img_e345 = b64(os.path.join(BASE, "blueprint_etapa345.png"))
img_e6   = b64(os.path.join(BASE, "blueprint_etapa6.png"))
img_e789 = b64(os.path.join(BASE, "blueprint_etapa789.png"))

# Estadísticas del dataset
metas    = [f for f in os.listdir(DS) if f.endswith("_metadata.json")] if os.path.exists(DS) else []
n_synth  = len([f for f in os.listdir(DS) if f.endswith(".png") and "_mask" not in f]) if os.path.exists(DS) else 0
sample   = {}
if metas:
    with open(os.path.join(DS, metas[0])) as f:
        sample = json.load(f)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>DigPatho – Pipeline de Generación Sintética (Reporte Completo)</title>
<style>
  body  {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.75; margin: 0; padding: 0; color: #222; }}
  .cover{{ background: #1a2e44; color: white; padding: 60px 80px; }}
  .cover h1{{ font-size: 2.4em; margin-bottom:10px; }}
  .cover p {{ font-size: 1.1em; opacity:.85; max-width:700px; }}
  .cover .badge{{ display:inline-block; background:#3ea6ff; border-radius:20px; padding:4px 16px; font-size:.85em; margin-top:14px; }}
  .content{{ max-width:1050px; margin:0 auto; padding:40px 30px; }}
  h2{{ color:#1a2e44; border-bottom:2px solid #1a2e44; padding-bottom:6px; margin-top:50px; font-size:1.4em; }}
  h3{{ color:#2c7a4b; margin-top:28px; }}
  img{{ max-width:100%; height:auto; border:1px solid #ddd; box-shadow:0 2px 10px rgba(0,0,0,.12); margin:16px 0; display:block; border-radius:4px; }}
  table{{ border-collapse:collapse; width:100%; margin:16px 0; }}
  th,td{{ border:1px solid #ddd; padding:10px 14px; text-align:left; font-size:13.5px; }}
  th{{ background:#f0f4f8; font-weight:600; }}
  .note{{ background:#e8f4fd; border-left:5px solid #2196F3; padding:14px 18px; margin:18px 0; border-radius:4px; }}
  .ok  {{ background:#e8f5e9; border-left:5px solid #4caf50; padding:14px 18px; margin:18px 0; border-radius:4px; }}
  .warn{{ background:#fff8e1; border-left:5px solid #ffc107; padding:14px 18px; margin:18px 0; border-radius:4px; }}
  .formula{{ background:#f5f5f5; border-left:4px solid #999; padding:10px 18px; font-family:monospace; margin:10px 0; font-size:13.5px; }}
  .stats{{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:22px 0; }}
  .card{{ background:#f0faf4; border:1px solid #b2dfdb; padding:18px; border-radius:8px; text-align:center; }}
  .card .val{{ font-size:2em; font-weight:700; color:#1a6e3c; }}
  .card .lbl{{ font-size:11.5px; color:#555; margin-top:4px; }}
  pre{{ background:#f5f5f5; padding:16px; border-radius:6px; font-size:12px; overflow-x:auto; }}
  .divider{{ border:none; border-top:1px solid #e0e0e0; margin:48px 0; }}
  @media print {{ .cover{{ padding:30px 40px; }} body{{ font-size:13px; }} }}
</style>
</head>
<body>

<div class="cover">
  <h1>DigPatho – Pipeline de Generación Sintética de Células HER2</h1>
  <p>Reporte técnico completo del pipeline de generación de datos sintéticos histopatológicos.
     Documenta las 9 etapas ejecutadas sobre imágenes reales de tejido HER2, desde la adquisición
     hasta la exportación del dataset de entrenamiento.</p>
  <div class="badge">Grado 3+ | Imagen JPEG real | Pipeline Completo</div>
</div>

<div class="content">

<!-- CONTEXTO -->
<h2>Contexto y Objetivo</h2>
<p>Este pipeline <strong>no</strong> clasifica HER2 ni diagnostica cáncer. Su única responsabilidad es tomar imágenes histopatológicas reales y generar variantes sintéticas que amplíen el dataset de entrenamiento del modelo clasificador de DigPatho.</p>
<div class="note">
  <strong>Hipótesis científica:</strong> Una generación sintética guiada por restricciones geométricas y fractales producirá datos de entrenamiento más representativos que los métodos tradicionales de augmentación (rotaciones, flips, ruido aleatorio).
</div>

<div class="stats">
  <div class="card"><div class="val">1</div><div class="lbl">Imagen fuente procesada (Grado 3+)</div></div>
  <div class="card"><div class="val">63</div><div class="lbl">Células reales extraídas y validadas</div></div>
  <div class="card"><div class="val">{n_synth}</div><div class="lbl">Células sintéticas generadas</div></div>
  <div class="card"><div class="val">9</div><div class="lbl">Etapas del Blueprint ejecutadas</div></div>
</div>

<hr class="divider">

<!-- ETAPAS 1 y 2 -->
<h2>Etapas 1 y 2 — Adquisición y Detección de Candidatos</h2>
<h3>Etapa 1: Sliding Window</h3>
<p>La imagen completa se recorre con una ventana de <strong>256×256 px</strong> y un <strong>overlap del 20%</strong>, garantizando que ninguna célula quede truncada en el borde de una ventana.</p>
<h3>Etapa 2: Heatmap de Probabilidad Híbrido</h3>
<p>Para cada ventana, un sistema de detección híbrido evalúa la probabilidad de que exista tejido celular válido combinando:</p>
<table>
  <tr><th>Método</th><th>Descripción</th><th>Peso</th></tr>
  <tr><td>Macenko – Canal Hematoxilina</td><td>Detecta núcleos celulares (azul)</td><td>45%</td></tr>
  <tr><td>Varianza local (textura)</td><td>Descarta regiones planas/vacías</td><td>30%</td></tr>
  <tr><td>Densidad de color</td><td>Filtra fondo blanco adiposo</td><td>15%</td></tr>
  <tr><td>Intensidad DAB</td><td>Señal HER2 en la membrana</td><td>10%</td></tr>
</table>
<p>Solo las regiones con puntaje <strong>P &gt; 0.60</strong> avanzan. Se obtienen centroide y bounding box de cada candidato y se aplica supresión de duplicados por overlap.</p>
<img src="data:image/png;base64,{img_e12}" alt="Mapa detección Etapas 1-2" />

<hr class="divider">

<!-- ETAPAS 3-4-5 -->
<h2>Etapas 3, 4 y 5 — Segmentación, Validación y Extracción</h2>
<h3>Etapa 3: SAM 2 con Prompt Focalizado</h3>
<p>Cada candidato es enviado a <strong>Segment Anything 2</strong> con un triple prompt para forzar que la máscara se ajuste a la membrana y no al borde del parche:</p>
<div class="formula">SAM 2 ( Imagen local  +  Centroide [Point]  +  Bounding Box )</div>

<h3>Etapa 4: Validación Geométrica de la Máscara</h3>
<table>
  <tr><th>Criterio</th><th>Umbral</th><th>Motivo</th></tr>
  <tr><td>Área</td><td>200 – 8000 px²</td><td>Descartar polvo y clusters</td></tr>
  <tr><td>Circularidad (4πA/P²)</td><td>&gt; 0.15</td><td>Descartar fibras alargadas</td></tr>
  <tr><td>Edge-Touch</td><td>Margen 2px</td><td>Descartar células truncadas</td></tr>
</table>

<h3>Etapa 5: Extracción RGBA</h3>
<p>Imagen RGB + canal Alpha (= máscara SAM 2) → célula aislada con fondo transparente. 89 candidatos → <strong>63 células puras</strong> superaron la validación.</p>
<img src="data:image/png;base64,{img_e345}" alt="Collage extracción RGBA Etapas 3-5" />

<hr class="divider">

<!-- ETAPA 6 -->
<h2>Etapa 6 — Caracterización Geométrica y Fractal</h2>
<p>Antes de generar ninguna variante sintética, se calcula el <strong>vector de características original</strong> de cada célula. Este vector actúa como restricción matemática en la Etapa 7.</p>

<h3>Dimensión Fractal D_f (Box-Counting)</h3>
<div class="formula">D_f = lím ( log N(ε) / log(1/ε) )</div>
<p>Mide la irregularidad topológica del borde de la membrana. Un contorno liso tiene D_f ≈ 1.0. Las membranas tumorales agresivas alcanzan D_f ≈ 1.4–1.7.</p>

<h3>Lacunaridad Λ</h3>
<div class="formula">Λ = (σ_M / μ_M)² + 1</div>
<p>Describe la heterogeneidad del espacio interno de la célula. Alta lacunaridad indica mayor agresividad tumoral.</p>

<h3>Intensidad DAB</h3>
<p>Concentración del reactivo DAB (marrón) extraída via Macenko. Alta concentración = sobreexpresión de HER2.</p>

<img src="data:image/png;base64,{img_e6}" alt="Distribuciones fractales Etapa 6" />

<hr class="divider">

<!-- ETAPAS 7-8-9 -->
<h2>Etapas 7, 8 y 9 — Generación Sintética, Validación y Dataset</h2>

<h3>Etapa 7: Generación con Restricciones Fractales</h3>
<table>
  <tr><th>Mecanismo</th><th>Qué muta</th><th>Restricción biológica</th></tr>
  <tr><td>fBm (Movimiento Browniano Fraccionario)</td><td>Borde/membrana</td><td>noise_strength inversamente proporcional a D_f original</td></tr>
  <tr><td>Macenko Augmentation</td><td>Intensidad H y DAB</td><td>Variación máx. ±30% de la concentración base</td></tr>
  <tr><td>TPS Warping (upscale ×4)</td><td>Píxeles internos</td><td>Alta resolución → sin artefactos blocky</td></tr>
</table>

<h3>Etapa 8: Validación de la Sintética</h3>
<div class="formula">SSIM &gt;= 0.25  →  conserva estructura interna</div>
<div class="formula">|D_f sintética − D_f original| &lt;= 0.30  →  morfología biológicamente coherente</div>

<h3>Etapa 9: Exportación del Dataset</h3>
<pre>
dataset_sintetico/
├── cell_XXXX_synNN.png           → RGBA (fondo transparente)
├── cell_XXXX_synNN_mask.png      → Máscara binaria SAM 2
└── cell_XXXX_synNN_metadata.json → Vector completo de características
</pre>

<img src="data:image/png;base64,{img_e789}" alt="Comparativo Original vs Sintética" />

<div class="ok">
  <strong>Resultado final:</strong> A partir de 1 imagen JPEG de grado 3+, el pipeline generó <strong>{n_synth} células sintéticas validadas</strong>, cada una con su máscara y su metadata completa de características geométricas y fractales.
</div>

<hr class="divider">

<!-- SIGUIENTES PASOS -->
<h2>Siguientes Pasos</h2>
<table>
  <tr><th>Paso</th><th>Descripción</th></tr>
  <tr><td>1</td><td>Correr el pipeline sobre todas las imágenes de los grados 0, 1+, 2+ y 3+</td></tr>
  <tr><td>2</td><td>Re-entrenar el clasificador ResNet50 de DigPatho con el dataset aumentado</td></tr>
  <tr><td>3</td><td>Comparar métricas (AUC, F1) entre el modelo entrenado con y sin sintéticas</td></tr>
  <tr><td>4</td><td>Implementar validación doble ciego con patólogos (real vs. sintética)</td></tr>
  <tr><td>5</td><td>Conectar con Grounding DINO para detección automática en WSI completas</td></tr>
</table>

<div class="warn">
  <strong>Limitación actual:</strong> El filtro de detección (Etapa 2) puede detectar puntos de tejido que no son células individuales. La precisión mejorará al conectar con las coordenadas del modelo principal de DigPatho o con Grounding DINO.
</div>

</div>
</body>
</html>"""

out = r"E:\Genaro\Desktop\Digital pathologies\no entrar\DigPatho_Pipeline_Completo.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Informe unificado: {out}")
