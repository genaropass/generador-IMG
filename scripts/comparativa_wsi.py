"""
Comparativa visual: Original vs Imagen Completa Sintetica con Células Mutadas (fBm)
Combina el fondo alterado por Macenko con las células que tienen mutaciones de forma.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from macenko import augment_staining

WSI_PATH = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts\3+\3+\3Her2.jpg"
DS_DIR   = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\dataset_sintetico"
OUT      = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\comparativa_wsi.png"

print("Cargando imagen original...")
img_pil = Image.open(WSI_PATH).convert("RGBA")
W, H = img_pil.size
img_np = np.array(img_pil.convert("RGB"))

print("Generando fondos aumentados (Macenko)...")
v1_bg = augment_staining(img_np, h_scale=0.75, dab_scale=1.30, brightness_scale=1.10)
v2_bg = augment_staining(img_np, h_scale=1.25, dab_scale=0.70, brightness_scale=0.90)
v3_bg = augment_staining(img_np, h_scale=0.90, dab_scale=1.15, brightness_scale=1.05)

# Convertir a PIL RGBA para poder pegar las celulas con alpha blend
v1_pil = Image.fromarray(v1_bg).convert("RGBA")
v2_pil = Image.fromarray(v2_bg).convert("RGBA")
v3_pil = Image.fromarray(v3_bg).convert("RGBA")

print("Pegando células sintéticas mutadas (fBm) en sus posiciones...")
metas = sorted([f for f in os.listdir(DS_DIR) if f.endswith("_metadata.json")])

for mf in metas:
    with open(os.path.join(DS_DIR, mf)) as f:
        meta = json.load(f)

    gx = int(meta.get("global_x", 0))
    gy = int(meta.get("global_y", 0))
    cell_id = meta.get("id", "")

    img_name = mf.replace("_metadata.json", ".png")
    img_path = os.path.join(DS_DIR, img_name)
    if not os.path.exists(img_path):
        continue

    cell = Image.open(img_path).convert("RGBA")
    cw, ch = cell.size
    if gx + cw > W or gy + ch > H or gx < 0 or gy < 0:
        continue

    # Pegar la célula en la variante correspondiente
    # Asumimos que syn00 va a v1, syn01 a v2, syn02 a v3
    if "syn00" in cell_id:
        v1_pil.paste(cell, (gx, gy), mask=cell)
    elif "syn01" in cell_id:
        v2_pil.paste(cell, (gx, gy), mask=cell)
    elif "syn02" in cell_id:
        v3_pil.paste(cell, (gx, gy), mask=cell)

print("Generando collage...")
fig, axes = plt.subplots(2, 2, figsize=(16, 15))

axes[0][0].imshow(img_pil.convert("RGB"));  axes[0][0].set_title("ORIGINAL", fontsize=14, fontweight='bold', color='black')
axes[0][1].imshow(v1_pil.convert("RGB"));   axes[0][1].set_title("Sintética 1 — DAB intenso (+30%) + fBm Mutación", fontsize=14, color='#8B0000')
axes[1][0].imshow(v2_pil.convert("RGB"));   axes[1][0].set_title("Sintética 2 — H oscuro, DAB pálido (-30%) + fBm Mutación", fontsize=14, color='#00008B')
axes[1][1].imshow(v3_pil.convert("RGB"));   axes[1][1].set_title("Sintética 3 — Intermedia + fBm Mutación", fontsize=14, color='#006400')

for row in axes:
    for ax in row:
        ax.axis("off")

plt.suptitle("DigPatho — WSI Original vs 3 Variantes Sintéticas Completas (Macenko + fBm)", 
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, facecolor='white', bbox_inches='tight')
print(f"Guardado: {OUT}")
