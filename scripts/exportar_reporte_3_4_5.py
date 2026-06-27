import base64
import os

img1_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\blueprint_etapa1_2.png"
img2_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\blueprint_etapa345.png"

# Incluir hasta 6 células individuales extraídas
cells_dir = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\extracted_cells"
cell_files = sorted([f for f in os.listdir(cells_dir) if f.endswith(".png") and "_mask" not in f])[:8]

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

b64_img1 = img_to_base64(img1_path)
b64_img2 = img_to_base64(img2_path)
cell_b64s = [img_to_base64(os.path.join(cells_dir, f)) for f in cell_files]

cell_gallery_html = "".join([
    f"""<div style="text-align:center; background:#111; padding:8px; border-radius:8px;">
        <img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto; max-height:120px;" />
        <div style="color:#aaa; font-size:11px; margin-top:4px;">{cell_files[i]}</div>
       </div>"""
    for i, b64 in enumerate(cell_b64s) if b64
])

html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DigPatho – Etapas 3, 4 y 5: Segmentación, Validación y Extracción</title>
<style>
    body {{ font-family: Arial, sans-serif; line-height: 1.7; margin: 40px auto; max-width: 1050px; color: #333; }}
    h1, h2, h3 {{ color: #2c3e50; }}
    h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
    h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 35px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 2px 2px 8px rgba(0,0,0,0.15); margin: 15px 0; display: block; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 11px; text-align: left; }}
    th {{ background-color: #f4f4f4; font-weight: bold; }}
    .note {{ background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 4px; }}
    .formula {{ background: #f9f9f9; border-left: 4px solid #888; padding: 10px 18px; font-family: monospace; margin: 12px 0; }}
    .gallery {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
    .stats-box {{ background: #f0faf0; border: 1px solid #b2dfdb; padding: 15px; border-radius: 6px; margin: 15px 0; }}
</style>
</head>
<body>

<h1>DigPatho – Etapas 3, 4 y 5: Segmentación Precisa, Validación Biológica y Extracción RGBA</h1>

<p>Este reporte documenta el procesamiento de los <strong>89 candidatos celulares</strong> identificados en las Etapas 1 y 2, a través de las tres etapas de refinamiento, validación y aislamiento.</p>

<h2>Etapa 3 – Segmentación Precisa con SAM 2</h2>
<p>Cada candidato fue remitido al modelo fundacional <strong>Segment Anything 2 (SAM 2)</strong> con un input triple: la imagen local, el centroide exacto del núcleo, y la bounding box calculada biológicamente. Este triple prompt elimina la ambigüedad del modelo y lo forza a ceñirse a la membrana celular.</p>
<table>
    <tr><th>Componente del Prompt</th><th>Origen</th><th>Propósito</th></tr>
    <tr><td>Imagen local recortada</td><td>WSI / Etapa 1</td><td>Contexto visual inmediato para el modelo</td></tr>
    <tr><td>Prompt Point (centroide)</td><td>Etapa 2 – Componentes Conectados</td><td>Señala al modelo qué objeto debe segmentar</td></tr>
    <tr><td>Bounding Box</td><td>Etapa 2 – Macenko/CC</td><td>Delimita el área máxima donde buscar la membrana</td></tr>
</table>

<h2>Etapa 4 – Validación Matemática de la Segmentación</h2>
<p>No todas las máscaras devueltas por SAM 2 son biológicamente válidas. Cada máscara pasó por tres filtros matemáticos rigurosos:</p>
<div class="formula">1. Área:  200 px² &lt; A &lt; 8000 px²</div>
<div class="formula">2. Circularidad:  C = (4π × Área) / Perímetro² &gt; 0.15</div>
<div class="formula">3. Edge-Touch: si la máscara toca los 2px del borde → DESCARTE</div>

<div class="stats-box">
    <strong>Resultados del filtro:</strong><br>
    89 candidatos ingresaron a la Etapa 4 → <strong>63 células puras aprobadas</strong> (29% descartadas por artefactos, cortes de borde o formas no celulares).
</div>

<h2>Etapa 5 – Extracción RGBA (Aislamiento Total)</h2>
<p>Las 63 células validadas fueron aisladas del tejido circundante. Se generaron tres archivos independientes por cada célula:</p>
<ul>
    <li><code>cell_XXXX.png</code> → Imagen RGBA con fondo completamente transparente (canal Alpha = máscara SAM 2).</li>
    <li><code>cell_XXXX_mask.png</code> → Máscara binaria pura para uso en entrenamiento de segmentación.</li>
</ul>

<h2>Collage de Células Extraídas (Muestra de 16)</h2>
<p>Visualización de una muestra de las células aisladas sobre fondo negro para evidenciar la transparencia del canal Alpha:</p>
<img src="data:image/png;base64,{b64_img2}" alt="Collage de células extraídas" />

<h2>Galería de Células Individuales (8 muestras)</h2>
<div class="gallery">
{cell_gallery_html}
</div>

<div class="note">
<strong>Siguiente paso – Etapa 6: Caracterización Geométrica y Fractal</strong><br>
Con 63 células aisladas perfectamente, el pipeline está listo para calcular el <strong>vector de características biológicas</strong> de cada una: Dimensión Fractal, Lacunaridad, Circularidad, Convexidad, Solidez e Intensidad de Tinción DAB (HER2). Este vector actuará como restricción matemática en la Etapa 7 (Generación Sintética Fractal), garantizando que las células artificiales mantengan propiedades biológicamente plausibles.
</div>

</body>
</html>
"""

output_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\Reporte_Etapa_3_4_5.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Reporte exportado: {output_path}")
