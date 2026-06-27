"""
Reconstruccion de Whole Slide Image Sintetica
Pega cada celula sintetica en su posicion original sobre la imagen base.
"""
import os, json
import numpy as np
from PIL import Image

WSI_PATH = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts\3+\3+\3Her2.jpg"
DS_DIR   = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\dataset_sintetico"
OUT_PATH = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\synthetic_wsi.png"

print("Cargando imagen original...")
wsi = Image.open(WSI_PATH).convert("RGBA")
W, H = wsi.size
print(f"Tamano WSI: {W}x{H}")

# Capa sintetica (misma resolucion que la WSI, fondo transparente)
synth_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

metas = sorted([f for f in os.listdir(DS_DIR) if f.endswith("_metadata.json")])
placed = 0

for mf in metas:
    with open(os.path.join(DS_DIR, mf)) as f:
        meta = json.load(f)

    gx = int(meta.get("global_x", 0))
    gy = int(meta.get("global_y", 0))

    # Imagen sintetica correspondiente
    img_name = mf.replace("_metadata.json", ".png")
    img_path = os.path.join(DS_DIR, img_name)
    if not os.path.exists(img_path):
        continue

    cell = Image.open(img_path).convert("RGBA")
    cw, ch = cell.size

    # Verificar que no se salga de los bordes
    if gx + cw > W or gy + ch > H or gx < 0 or gy < 0:
        continue

    # Pegar con blend por alpha
    synth_layer.paste(cell, (gx, gy), mask=cell)
    placed += 1

# Componer: WSI original + capa sintetica encima
result = Image.alpha_composite(wsi, synth_layer)
result = result.convert("RGB")
result.save(OUT_PATH, quality=95)

print(f"Celulas colocadas: {placed}")
print(f"WSI sintetica guardada en: {OUT_PATH}")
