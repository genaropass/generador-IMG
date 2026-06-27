import base64, os, json

img_path  = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\blueprint_etapa789.png"
ds_dir    = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\dataset_sintetico"

def b64(p):
    if os.path.exists(p):
        with open(p,"rb") as f: return base64.b64encode(f.read()).decode()
    return ""

chart_b64 = b64(img_path)

# Contar dataset
all_meta  = [f for f in os.listdir(ds_dir) if f.endswith("_metadata.json")]
accepted  = len([f for f in os.listdir(ds_dir) if f.endswith(".png") and "_mask" not in f])

# Leer métricas del primer meta disponible como ejemplo
sample_meta = {}
if all_meta:
    with open(os.path.join(ds_dir, all_meta[0])) as f:
        sample_meta = json.load(f)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DigPatho – Etapas 7, 8 y 9: Generación Sintética, Validación y Dataset</title>
<style>
    body{{ font-family:Arial,sans-serif; line-height:1.7; margin:40px auto; max-width:1050px; color:#333; }}
    h1,h2,h3{{ color:#2c3e50; }}
    h1{{ border-bottom:3px solid #2c3e50; padding-bottom:10px; }}
    h2{{ border-bottom:1px solid #ccc; padding-bottom:5px; margin-top:35px; }}
    img{{ max-width:100%; height:auto; border:1px solid #ddd; box-shadow:2px 2px 8px rgba(0,0,0,.15); margin:15px 0; display:block; }}
    .formula{{ background:#f9f9f9; border-left:4px solid #888; padding:10px 18px; font-family:monospace; margin:12px 0; }}
    .note{{ background:#e7f3fe; border-left:6px solid #2196F3; padding:15px; margin:20px 0; border-radius:4px; }}
    .success{{ background:#e8f5e9; border-left:6px solid #4caf50; padding:15px; margin:20px 0; border-radius:4px; }}
    .stats{{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:20px 0; }}
    .card{{ background:#f0faf0; border:1px solid #b2dfdb; padding:16px; border-radius:8px; text-align:center; }}
    .card .val{{ font-size:28px; font-weight:bold; color:#2c7a4b; }}
    .card .label{{ font-size:12px; color:#666; margin-top:4px; }}
    pre{{ background:#f5f5f5; padding:15px; border-radius:6px; font-size:12px; overflow-x:auto; }}
    table{{ border-collapse:collapse; width:100%; margin:20px 0; }}
    th,td{{ border:1px solid #ddd; padding:10px; text-align:left; font-size:13px; }}
    th{{ background:#f4f4f4; }}
</style>
</head>
<body>

<h1>DigPatho – Etapas 7, 8 y 9: Generación Sintética, Validación y Dataset Final</h1>

<p>Estas etapas constituyen la fase de producción del pipeline. El sistema genera variantes sintéticas biológicamente plausibles de cada célula extraída, las valida matemáticamente y las exporta como dataset estructurado listo para entrenar modelos de IA.</p>

<div class="stats">
    <div class="card"><div class="val">63</div><div class="label">Células reales procesadas</div></div>
    <div class="card"><div class="val">125</div><div class="label">Sintéticas aceptadas</div></div>
    <div class="card"><div class="val">64</div><div class="label">Rechazadas por validación</div></div>
    <div class="card"><div class="val">66%</div><div class="label">Tasa de aceptación</div></div>
</div>

<h2>Etapa 7 – Generación Sintética Guiada por Restricciones Fractales</h2>
<p>Cada célula real fue mutada mediante dos mecanismos complementarios, ambos acotados por el vector de características calculado en la Etapa 6:</p>
<table>
    <tr><th>Mecanismo</th><th>Qué muta</th><th>Restricción biológica</th></tr>
    <tr><td>Movimiento Browniano Fraccionario (fBm)</td><td>Forma del borde / membrana</td><td>noise_strength inversamente proporcional a D_f original</td></tr>
    <tr><td>Macenko Augmentation</td><td>Intensidad H y DAB (color)</td><td>Variación máx. ±30% respecto a la concentración original</td></tr>
    <tr><td>TPS Warping</td><td>Píxeles internos</td><td>Se adaptan al nuevo contorno manteniendo textura</td></tr>
</table>

<h2>Etapa 8 – Validación Matemática de Cada Sintética</h2>
<p>Antes de aceptar una célula sintética en el dataset, pasa por dos filtros automáticos:</p>
<div class="formula">SSIM (Structural Similarity Index) >= 0.25  →  la sintética conserva la estructura interna</div>
<div class="formula">|D_f sintética - D_f original|  <=  0.30  →  la morfología fractal es biológicamente coherente</div>
<p>Las 64 células rechazadas fallaron alguno de estos umbrales (demasiado distorsionadas o con D_f aberrante).</p>

<h2>Collage Comparativo: Original vs Sintética</h2>
<img src="data:image/png;base64,{chart_b64}" alt="Collage Original vs Sintetica" />

<h2>Etapa 9 – Estructura del Dataset Exportado</h2>
<p>Cada muestra sintética aceptada se almacenó en tres archivos dentro de <code>dataset_sintetico/</code>:</p>
<pre>
dataset_sintetico/
├── cell_0001_syn00.png           → Imagen RGBA (fondo transparente)
├── cell_0001_syn00_mask.png      → Máscara binaria SAM 2
├── cell_0001_syn00_metadata.json → Vector completo de características
├── cell_0001_syn01.png
├── ...
└── cell_0080_syn02_metadata.json
</pre>

<h3>Ejemplo de Metadata JSON</h3>
<pre>{json.dumps(sample_meta, indent=2, ensure_ascii=False)[:800]}...</pre>

<div class="success">
<strong>Pipeline DigPatho Completado</strong><br>
A partir de <strong>1 imagen JPEG de grado 3+</strong> provista por el patólogo, el pipeline completo ejecutó las 9 etapas del Blueprint Científico y produjo un dataset de <strong>125 células sintéticas validadas</strong>, cada una con su máscara binaria y su metadata completa de características geométricas y fractales. Este dataset está listo para ser usado en el entrenamiento del clasificador HER2 de DigPatho.
</div>

</body>
</html>"""

out = r"E:\Genaro\Desktop\Digital pathologies\no entrar\Reporte_Etapa_7_8_9.html"
with open(out,"w",encoding="utf-8") as f: f.write(html)
print(f"Exportado: {out}")
