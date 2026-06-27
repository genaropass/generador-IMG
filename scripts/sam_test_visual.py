"""
Test visual minimo: SAM 2 sobre un punto exacto de la imagen.
Objetivo: ver QUE segmenta SAM cuando le apunto a un nucleo real.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import SAM

IMAGE = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts\3+\3+\3Her2.jpg"
OUT   = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\sam_test.png"

# Coordenadas de un nucleo azul visible (ajustar segun imagen)
# x=480, y=265 apunta al nucleo azul central de la imagen
PUNTO_X, PUNTO_Y = 480, 265

img = np.array(Image.open(IMAGE).convert("RGB"))
model = SAM("sam2.1_t.pt")

results = model.predict(
    source=img,
    points=[[PUNTO_X, PUNTO_Y]],
    labels=[1],
    verbose=False
)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(img)
axes[0].plot(PUNTO_X, PUNTO_Y, 'r*', markersize=18)
axes[0].set_title("Imagen original + Punto de interés", fontsize=13)
axes[0].axis("off")

axes[1].imshow(img)
if results and results[0].masks and len(results[0].masks.data) > 0:
    mask = results[0].masks.data[0].cpu().numpy()
    if mask.shape != img.shape[:2]:
        import cv2
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    axes[1].imshow(mask, alpha=0.5, cmap='Reds')
    axes[1].set_title("SAM 2: mascara detectada", fontsize=13)
else:
    axes[1].set_title("SAM 2: sin mascara", fontsize=13)
axes[1].plot(PUNTO_X, PUNTO_Y, 'r*', markersize=18)
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUT, dpi=150, facecolor='white')
print(f"Guardado: {OUT}")
