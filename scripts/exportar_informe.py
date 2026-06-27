import base64
import os

img1_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\pipeline_paso_a_paso.png"
img2_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\todas_las_variantes.png"

def img_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

b64_img1 = img_to_base64(img1_path)
b64_img2 = img_to_base64(img2_path)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Informe: Pipeline de Generación Sintética</title>
<style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px auto; max-width: 1000px; color: #333; }}
    h1, h2, h3 {{ color: #2c3e50; }}
    h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
    h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin: 20px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
    th {{ background-color: #f8f9fa; font-weight: bold; }}
    .note {{ background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 15px; margin: 20px 0; }}
</style>
</head>
<body>

<h1>Informe: Pipeline de Generación Sintética de Células HER2</h1>

<p>Este informe detalla el proceso y los resultados obtenidos al aplicar el nuevo <strong>Pipeline de Generación Sintética de 3 Capas</strong> sobre imágenes reales obtenidas de muestras patológicas (Grado 3+). El objetivo de este pipeline es generar un gran volumen de datos artificiales pero biológicamente plausibles para entrenar nuestro modelo de Inteligencia Artificial (ResNet50) de manera más robusta, resolviendo el problema de la falta de datos de entrenamiento.</p>

<h2>1. El Desafío</h2>
<p>Al entrenar modelos de Inteligencia Artificial en patología digital, a menudo enfrentamos:</p>
<ol>
    <li><strong>Escasez de datos:</strong> Muy pocas imágenes correctamente anotadas.</li>
    <li><strong>Sobreajuste (Overfitting):</strong> La IA aprende a reconocer el "color" de un laboratorio específico o la iluminación de un escáner, en lugar de aprender la forma real (morfología) del cáncer.</li>
</ol>
<p>Para solucionar esto, hemos construido una herramienta que toma una sola célula real y genera decenas de "células gemelas" sintéticas. Estas nuevas células tienen formas distintas (usando geometría fractal) y colores distintos, forzando a la IA a aprender verdaderos patrones clínicos.</p>

<h2>2. El Pipeline de 3 Capas</h2>
<p>Hemos desarrollado tres módulos de software que actúan en secuencia sobre la imagen real:</p>

<h3>Capa 1: El Motor Fractal (Morfología)</h3>
<p>En la naturaleza, los tumores y las membranas celulares crecen siguiendo patrones matemáticos llamados <strong>fractales</strong>. En lugar de deformar la célula estirándola al azar, aplicamos una ecuación matemática (Movimiento Browniano Fraccionario) al contorno de la célula. Esto le añade "rugosidades" y "dientes" realistas a la membrana, simulando células tumorales más o menos agresivas.</p>

<h3>Capa 2: Algoritmo de Macenko (Tinción y Color)</h3>
<p>En las muestras HER2 usamos dos colores: Hematoxilina (azul para los núcleos) y DAB (marrón para la membrana). Esta capa descompone matemáticamente la imagen en esos dos colores puros. Una vez separados, altera sutilmente sus concentraciones (hace el marrón más oscuro, el azul más claro, cambia el brillo) y los vuelve a unir. Esto simula diferentes laboratorios y microscopios.</p>

<h3>Capa 3: Warping TPS (Ajuste Final de Píxeles)</h3>
<p>Finalmente, usamos un algoritmo geométrico (Thin Plate Spline) para que la foto real de la célula se "estire" fluidamente y encaje a la perfección dentro del nuevo contorno fractal creado en la Capa 1, manteniendo la textura biológica intacta.</p>

<h2>3. Resultados Visuales sobre Imagen Real (Grado 3+)</h2>
<p>Tomamos una imagen real del grado 3+ (<code>3Her2.jpg</code>) y aplicamos el pipeline.</p>

<h3>El Proceso Paso a Paso</h3>
<p>A continuación se muestra cómo se transforma la imagen a medida que pasa por las capas:</p>

<img src="data:image/png;base64,{b64_img1}" alt="Pipeline Paso a Paso" />

<table>
    <tr><th>Paso</th><th>Descripción</th></tr>
    <tr><td><strong>Original</strong></td><td>Parche de 220x220px extraído del centro de la imagen real del patólogo.</td></tr>
    <tr><td><strong>Capa 2: Macenko</strong></td><td>Los mismos píxeles con la concentración de H y DAB matemáticamente alterada.</td></tr>
    <tr><td><strong>Capa 1: Fractal</strong></td><td>El contorno verde original deformado por fBm en un nuevo borde rojo fractal.</td></tr>
    <tr><td><strong>Capa 3: TPS + Final</strong></td><td>Los píxeles reales adaptados matemáticamente a la nueva morfología fractal.</td></tr>
</table>

<h3>Generación Masiva (4 Variantes)</h3>
<p>A partir de la misma imagen original, el sistema generó automáticamente 4 variaciones únicas:</p>

<img src="data:image/png;base64,{b64_img2}" alt="Todas las Variantes" />

<div class="note">
<strong>Como se puede observar:</strong>
<ul>
    <li>Cada célula sintética tiene un borde único (morfología fractal alterada).</li>
    <li>Cada célula sintética tiene una intensidad de color única (simulando distintos laboratorios).</li>
    <li>Todas conservan la textura y apariencia de un tejido biológico real.</li>
</ul>
</div>

<h2>4. Conclusión y Siguientes Pasos</h2>
<p>El pipeline funcional demuestra que es posible <strong>multiplicar exponencialmente</strong> nuestro dataset de entrenamiento sin necesidad de etiquetar manualmente miles de imágenes. Al entrenar a la Inteligencia Artificial con estas imágenes fractales, el modelo se volverá invulnerable a cambios de laboratorio y aprenderá a clasificar basándose en la verdadera topología del tumor.</p>

<p>El siguiente paso es automatizar este script para que recorra todas las imágenes de los grados 0, 1+, 2+ y 3+, generando la base de datos sintética final para el entrenamiento.</p>

</body>
</html>
"""

output_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\Informe_Generacion_Sintetica.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Informe generado en: {output_path}")
