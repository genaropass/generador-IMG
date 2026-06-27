import base64, os, json

img_path   = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\blueprint_etapa6.png"
json_path  = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\feature_vectors.json"

def img_to_b64(path):
    if os.path.exists(path):
        with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
    return ""

b64_chart = img_to_b64(img_path)

with open(json_path, encoding="utf-8") as f:
    vectors = json.load(f)

# Tabla con las primeras 12 células
table_rows = ""
for v in vectors[:12]:
    geo   = v["geometric"]
    df    = v["fractal_dimension"]
    lac   = v["lacunarity"]
    dab   = v["staining"].get("dab_mean", 0)
    table_rows += f"""<tr>
        <td>{v['id']}</td>
        <td>{int(geo['area_px'])}</td>
        <td>{geo['circularity']:.3f}</td>
        <td>{geo['solidity']:.3f}</td>
        <td><strong>{df:.4f}</strong></td>
        <td>{'N/A' if str(lac)=='nan' else f'{float(lac):.4f}'}</td>
        <td>{max(0,dab):.4f}</td>
    </tr>"""

html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DigPatho – Etapa 6: Caracterización Geométrica y Fractal</title>
<style>
    body{{ font-family:Arial,sans-serif; line-height:1.7; margin:40px auto; max-width:1050px; color:#333; }}
    h1,h2,h3{{ color:#2c3e50; }}
    h1{{ border-bottom:3px solid #2c3e50; padding-bottom:10px; }}
    h2{{ border-bottom:1px solid #ccc; padding-bottom:5px; margin-top:35px; }}
    img{{ max-width:100%; height:auto; border:1px solid #ddd; box-shadow:2px 2px 8px rgba(0,0,0,.15); margin:15px 0; display:block; }}
    table{{ border-collapse:collapse; width:100%; margin:20px 0; }}
    th,td{{ border:1px solid #ddd; padding:10px; text-align:left; font-size:13px; }}
    th{{ background:#f4f4f4; font-weight:bold; }}
    .note{{ background:#e7f3fe; border-left:6px solid #2196F3; padding:15px; margin:20px 0; border-radius:4px; }}
    .formula{{ background:#f9f9f9; border-left:4px solid #888; padding:10px 18px; font-family:monospace; margin:12px 0; font-size:14px; }}
    .stats{{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:20px 0; }}
    .stat-card{{ background:#f0faf0; border:1px solid #b2dfdb; padding:16px; border-radius:8px; text-align:center; }}
    .stat-card .val{{ font-size:26px; font-weight:bold; color:#2c7a4b; }}
    .stat-card .label{{ font-size:12px; color:#666; margin-top:4px; }}
</style>
</head>
<body>

<h1>DigPatho – Etapa 6: Caracterización Geométrica y Fractal</h1>
<p>Esta etapa constituye el corazón científico del pipeline. Antes de cualquier modificación, cada célula extraída es <strong>caracterizada matemáticamente</strong> mediante un vector de descriptores que actuará como restricción biológica en la Generación Sintética (Etapa 7).</p>

<div class="stats">
    <div class="stat-card"><div class="val">{len(vectors)}</div><div class="label">Células Caracterizadas</div></div>
    <div class="stat-card"><div class="val">{sum(1 for v in vectors if v['fractal_dimension']>1.2)}</div><div class="label">Con D_f &gt; 1.2 (alta irregularidad)</div></div>
    <div class="stat-card"><div class="val">{sum(1 for v in vectors if v['staining'].get('dab_mean',0)>0.5)}</div><div class="label">Alta intensidad HER2 (DAB &gt; 0.5)</div></div>
    <div class="stat-card"><div class="val">{len([v for v in vectors if str(v['lacunarity'])!='nan'])}</div><div class="label">Con Lacunaridad calculada</div></div>
</div>

<h2>Los 3 Pilares de la Caracterización</h2>

<h3>1. Dimensión Fractal D_f (Box-Counting)</h3>
<p>Mide la irregularidad topológica del borde de la membrana celular. Un contorno perfectamente liso tiene D_f ≈ 1.0. Membranas tumorales agresivas alcanzan D_f ≈ 1.4–1.7.</p>
<div class="formula">D_f = lím ( log N(ε) / log(1/ε) )</div>
<p>donde N(ε) es el número de cajas de lado ε que contienen parte del contorno celular.</p>

<h3>2. Lacunaridad (Λ)</h3>
<p>Describe la heterogeneidad del espacio interno de la célula. Una célula homogénea tiene Λ ≈ 1.0. Células con cromatina irregular y espacios vacíos internos alcanzan Λ > 2.</p>
<div class="formula">Λ = (σ_M / μ_M)² + 1</div>

<h3>3. Intensidad DAB (Marcador HER2)</h3>
<p>Extraída mediante el algoritmo de Macenko. Cuantifica la concentración del reactivo DAB (color marrón) en la membrana de la célula. Alta concentración indica sobreexpresión de HER2 (Grado 3+).</p>

<h2>Panel de Distribuciones (63 Células – Grado 3+)</h2>
<img src="data:image/png;base64,{b64_chart}" alt="Distribuciones Fractales" />

<h2>Muestra del Vector de Características (Primeras 12 Células)</h2>
<table>
    <tr><th>ID</th><th>Área (px²)</th><th>Circularidad</th><th>Solidez</th><th>D_f (Fractal)</th><th>Lacunaridad</th><th>DAB (HER2)</th></tr>
    {table_rows}
</table>

<div class="note">
<strong>Próximo Paso – Etapa 7: Generación Sintética Guiada por Restricciones Fractales</strong><br>
Con los vectores de cada célula almacenados en <code>feature_vectors.json</code>, el pipeline procederá a la generación de variantes sintéticas. Las mutaciones de membrana (fBm) y de tinción (Macenko) respetarán estrictamente los rangos estadísticos de D_f y Λ medidos, garantizando la plausibilidad biológica de cada nueva célula artificial generada.
</div>

</body>
</html>
"""

output_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\Reporte_Etapa_6.html"
with open(output_path,"w",encoding="utf-8") as f:
    f.write(html_content)

print(f"Reporte exportado: {output_path}")
