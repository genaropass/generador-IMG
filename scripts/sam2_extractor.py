import os
import cv2
import numpy as np
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    from ultralytics import SAM
except ImportError:
    print("Por favor instala ultralytics: pip install ultralytics")
    SAM = None

class SAM2Extractor:
    def __init__(self, model_name="sam2.1_t.pt"):
        """
        Inicializa el modelo SAM 2. 
        Por defecto usa la version 'tiny' (sam2.1_t.pt) para velocidad en CPU/GPU ligera.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        if SAM is None:
            raise ImportError("Ultralytics no esta instalado.")
        print(f"Cargando modelo SAM 2: {self.model_name}...")
        self.model = SAM(self.model_name)
        print("Modelo SAM 2 cargado correctamente.")

    def extract_contour(self, image_rgb, center_point=None):
        """
        Extrae el contorno real de la celula en la imagen.
        
        Parametros:
            image_rgb: Imagen numpy array (H, W, 3).
            center_point: Tupla (x, y) de la coordenada a apuntar. Si es None, usa el centro de la imagen.
        
        Retorna:
            contour: Array numpy (N, 2) de coordenadas (y, x) del borde de la membrana.
        """
        H, W = image_rgb.shape[:2]
        
        if center_point is None:
            # Si no hay punto, apuntamos al centro exacto de la imagen (ideal para nuestros parches)
            center_point = [W // 2, H // 2]
            
        print(f"Inferencia SAM 2 en punto (x={center_point[0]}, y={center_point[1]})...")
        
        # Ejecutar modelo SAM
        # points: Lista de puntos
        # labels: 1 = objeto principal, 0 = fondo
        results = self.model.predict(
            source=image_rgb, 
            points=[center_point], 
            labels=[1], 
            verbose=False
        )
        
        if not results or results[0].masks is None:
            raise ValueError("SAM 2 no pudo encontrar ninguna celula en ese punto.")
            
        # Extraer la mascara binaria (0s y 1s)
        mask = results[0].masks.data[0].cpu().numpy()
        
        # Redimensionar la mascara al tamano original de la imagen 
        # (SAM suele procesar a 1024x1024 internamente y devuelve la mascara en esa escala a veces)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Extraer contorno matematico usando OpenCV
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("SAM 2 devolvio mascara vacia. No se detecto el borde.")
            
        # Tomar el contorno mas grande (por si detecto varias piezas pequeñas)
        main_contour = max(contours, key=cv2.contourArea)
        
        # main_contour tiene shape (N, 1, 2) con formato (x, y). 
        # Lo pasamos a formato (N, 2) en orden (y, x) para que sea compatible con nuestro pipeline TPS/Fractal
        contour_xy = main_contour.reshape(-1, 2)
        contour_yx = np.column_stack([contour_xy[:, 1], contour_xy[:, 0]])
        
        # Opcional: Suavizar un poco el contorno real original para evitar "ruido de pixel" de la mascara
        # y que el fractal trabaje sobre una curva biologica continua.
        # Simplificamos la curva con Douglas-Peucker (epsilon=1.0)
        epsilon = 0.1 * cv2.arcLength(main_contour, True) / 100.0
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        approx_xy = approx.reshape(-1, 2)
        contour_yx = np.column_stack([approx_xy[:, 1], approx_xy[:, 0]])
        
        return contour_yx


if __name__ == "__main__":
    # Prueba rapida y unitaria
    from PIL import Image
    import matplotlib.pyplot as plt
    
    img_path = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts\3+\3+\3Her2.jpg"
    img_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img_pil)
    
    # Recortar parche para la prueba
    patch = img_rgb[400:650, 400:650]
    
    extractor = SAM2Extractor()
    contorno = extractor.extract_contour(patch)
    
    plt.imshow(patch)
    # Contorno devuelto es (y, x), lo invertimos a (x, y) solo para matplotlib
    plt.plot(contorno[:, 1], contorno[:, 0], 'r-', lw=2)
    plt.title("Prueba Unitaria SAM 2")
    
    out_dir = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "prueba_unitaria_sam2.png"), dpi=200)
    print("Prueba completada.")
