import numpy as np
import matplotlib.pyplot as plt
import os

def generate_fractal_noise(angles, octaves=4, persistence=0.5, lacunarity=2.0, base_frequency=2.0):
    """
    Genera Ruido Fractal (fBm) usando suma de ondas senoidales.
    """
    noise = np.zeros_like(angles)
    amplitude = 1.0
    frequency = base_frequency
    
    for _ in range(octaves):
        phase = np.random.uniform(0, 2 * np.pi)
        noise += amplitude * np.sin(frequency * angles + phase)
        amplitude *= persistence
        frequency *= lacunarity
        
    return noise

def apply_fractal_mutation(contour, noise_strength=5.0, octaves=4, persistence=0.5, lacunarity=2.0, base_frequency=2.0):
    """
    Aplica la mutacion fractal a un contorno (array Nx2).
    """
    centroid = np.mean(contour, axis=0)
    
    shifted = contour - centroid
    angles = np.arctan2(shifted[:, 1], shifted[:, 0])
    radii = np.hypot(shifted[:, 0], shifted[:, 1])
    
    fractal_noise = generate_fractal_noise(angles, octaves, persistence, lacunarity, base_frequency)
    
    new_radii = radii + (fractal_noise * noise_strength)
    new_radii = np.maximum(new_radii, 1.0)
    
    # Para asegurar que la curva se cierre bien, aplicamos un suavizado en los extremos del array polar
    # Dado que los angulos se calculan de -pi a pi, al juntarse puede haber un salto abrupto.
    # En un entorno real SAM2 da bordes ordenados, pero aqui corregimos el posible gap:
    
    new_x = centroid[0] + new_radii * np.cos(angles)
    new_y = centroid[1] + new_radii * np.sin(angles)
    
    return np.column_stack((new_x, new_y))

def generate_mock_cell(num_points=300, a=50, b=40):
    """
    Genera una elipse pura simulando el contorno de una célula sana.
    """
    theta = np.linspace(-np.pi, np.pi, num_points)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.column_stack((x, y))

if __name__ == "__main__":
    print("Iniciando prueba del Motor Fractal...")
    
    # 1. Generar contorno sano (Mock)
    original_cell = generate_mock_cell()
    
    # 2. Generar diferentes mutaciones (Niveles de agresividad)
    mutacion_leve = apply_fractal_mutation(original_cell, noise_strength=2.0, octaves=2, persistence=0.3, base_frequency=3.0)
    mutacion_mod = apply_fractal_mutation(original_cell, noise_strength=4.0, octaves=4, persistence=0.5, base_frequency=4.0)
    mutacion_agresiva = apply_fractal_mutation(original_cell, noise_strength=7.0, octaves=6, persistence=0.7, base_frequency=6.0)
    
    # 3. Visualización con Matplotlib
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(original_cell[:, 0], original_cell[:, 1], 'g-')
    plt.fill(original_cell[:, 0], original_cell[:, 1], color='green', alpha=0.2)
    plt.title("Célula Sana (Contorno Real)")
    plt.axis('equal')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.plot(mutacion_leve[:, 0], mutacion_leve[:, 1], 'y-')
    plt.fill(mutacion_leve[:, 0], mutacion_leve[:, 1], color='yellow', alpha=0.3)
    plt.title("Mutación Leve\n(Fractal Básico)")
    plt.axis('equal')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.plot(mutacion_mod[:, 0], mutacion_mod[:, 1], 'orange')
    plt.fill(mutacion_mod[:, 0], mutacion_mod[:, 1], color='orange', alpha=0.4)
    plt.title("Mutación Moderada\n(Textura Tumoral)")
    plt.axis('equal')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.plot(mutacion_agresiva[:, 0], mutacion_agresiva[:, 1], 'r-')
    plt.fill(mutacion_agresiva[:, 0], mutacion_agresiva[:, 1], color='red', alpha=0.5)
    plt.title("Mutación Agresiva\n(Alta Dimensión Fractal)")
    plt.axis('equal')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "prueba_matematica_fractal.png")
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Prueba completada. Gráfico guardado en: {output_path}")
