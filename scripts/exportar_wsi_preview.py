import base64
import os

img_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\comparativa_wsi.png"

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

b64_img = img_to_base64(img_path)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DigPatho – Comparativa Whole Slide Image Sintética</title>
<style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px auto; max-width: 1000px; color: #333; }}
    h1, h2, h3 {{ color: #2c3e50; }}
    h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin: 20px 0; }}
    .note {{ background-color: #f9f9f9; border-left: 6px solid #4CAF50; padding: 15px; margin: 20px 0; }}
    ul {{ margin-top: 5px; }}
    li {{ margin-bottom: 10px; }}
</style>
</head>
<body>

<h1>Reporte DigPatho: Comparativa Whole Slide Image Sintética</h1>

<p>Este documento muestra la imagen original a nivel completo comparada con tres variaciones sintéticas generadas mediante alteraciones algorítmicas de la tinción (Macenko). Esta técnica asegura que el modelo de IA aprenda características independientes de factores externos como el escáner o las diferencias en la concentración de los reactivos de laboratorio.</p>

<img src="data:image/png;base64,{b64_img}" alt="Comparativa WSI Original y Sintéticas" />

<div class="note">
    <h3>Análisis de las Variantes</h3>
    <ul>
        <li><strong>Sintética 1 (DAB Intenso):</strong> Se ha intensificado la tinción DAB (marrón) en un 30%. Esto simula muestras con mayor expresión aparente de HER2, forzando a la red neuronal a no sobrediagnosticar falsos positivos solo por la intensidad del color.</li>
        <li><strong>Sintética 2 (H Oscuro, DAB Pálido):</strong> La Hematoxilina (azul de los núcleos) se ha oscurecido y el DAB se ha vuelto un 30% más pálido. Este efecto simula variaciones típicas que ocurren entre distintos laboratorios o por envejecimiento de los reactivos.</li>
        <li><strong>Sintética 3 (Variación Intermedia):</strong> Una variación más sutil y realista, combinando un leve cambio en ambos canales y el brillo global de la captura, replicando la iluminación de otro tipo de escáner digital.</li>
    </ul>
    <p><em>Al entrenar con todas estas variaciones simultáneamente, el clasificador de DigPatho se volverá altamente robusto frente a fluctuaciones en el proceso de adquisición de imágenes.</em></p>
</div>

</body>
</html>
"""

output_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\Reporte_WSI_Sintetica_Completa.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Reporte WSI exportado exitosamente a: {output_path}")
