# DigPatho — Pipeline Sintético v2

Generador de imágenes histopatológicas sintéticas mediante deformación elástica fractal y augmentación de tinción.

## Requisitos e Instalación

### 1. Activar el entorno virtual

El entorno virtual (`venv`) está dentro de `generador/`. Desde esa carpeta, actívalo así:

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```z
z
**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

Si aún no existe el entorno, créalo dentro de `generador/`:
```bash
python -m venv venv
```

### 2. Instalar dependencias

Con el entorno activado, instala las dependencias:
```bash
pip install -r requirements.txt
```

---

## Archivos del Proyecto

```text
generador/
├── scripts/
│   ├── app.py                    # Interfaz web (Streamlit)
│   ├── synthetic_pipeline.py     # Pipeline principal
│   ├── fractal_field.py          # Motor matemático fractal
│   ├── macenko.py                # Aumentación de color
│   ├── fractal_characterizer.py  # Box-Counting (Dimensión Fractal)
│   └── sam2.1_t.pt               # Pesos del modelo SAM 2
├── requirements.txt
└── README.md
```

---

## Uso — Interfaz Web (Recomendado)

1. Abre una terminal y muévete a la carpeta `scripts`:
```bash
cd generador/scripts
```
2. Inicia la interfaz gráfica:
```bash
python -m streamlit run app.py
```
3. Abre en tu navegador `http://localhost:8501`. Sube una imagen, ajusta la intensidad de deformación y haz clic en "Generar Dataset".

---

## Uso — Línea de Comandos

Ideal para procesar imágenes masivamente sin interfaz:
```bash
cd generador/scripts
python synthetic_pipeline.py
```
*(Para cambiar la imagen de entrada, edita las últimas líneas de `synthetic_pipeline.py`)*

---

## Estructura del Dataset Generado

Las imágenes se guardan automáticamente en `generador/scripts/output_sintetico_v2/dataset_sintetico/`. Cada variante incluye:
- `imagen_syn00.png` → La imagen WSI deformada limpia.
- `imagen_syn00_labels.csv` → Las coordenadas (X, Y) actualizadas de las células.
- `imagen_syn00_warpfield.npz` → El campo de deformación matemático (para reproducibilidad).

> **Nota:** El archivo `_labels.csv` está listo para usarse en el entrenamiento de detectores como YOLO o Faster R-CNN. Solo debes añadir tu grado clínico (Ej: HER2 3+).
