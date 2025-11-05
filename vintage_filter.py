import cv2
import numpy as np
import sys
from pathlib import Path

# Matriz 3x3 que remapea los canales BGR para darle un tono "vintage".
# Cada fila determina cómo se mezclan los canales originales para producir
# un nuevo valor para B, G y R respectivamente. Los valores se han elegido
# para calentar ligeramente los rojos y atenuar azules, típico en filtros vintage.
VINTAGE_MAT = np.array([
    [0.8, 0.3, 0.1],  # B <- mezcla de B,G,R
    [0.2, 0.7, 0.1],  # G <- mezcla de B,G,R
    [0.1, 0.3, 0.5],  # R <- mezcla de B,G,R
], dtype=np.float32)

def apply_gamma(img, gamma=1.07):
    """
    Aplica corrección gamma a la imagen usando una LUT para eficiencia.
    - gamma > 1 aclara la imagen ligeramente en este caso porque se aplica
      la potencia 1/gamma sobre el valor normalizado.
    Motivo: pequeños ajustes de gamma ayudan a obtener ese contraste y tono
    suave característico de filtros retro.
    """
    inv = 1.0 / gamma
    # LUT precalculada para transformar cada posible valor [0..255]
    lut = ((np.arange(256) / 255.0) ** inv * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)

def vintage(img):
    """
    Aplica el efecto vintage sobre una imagen BGR:
    1) Transformación de color lineal mediante VINTAGE_MAT.
    2) Aplicación de viñeteado radial para oscurecer bordes.
    3) Clipping y conversión a uint8.
    4) Corrección gamma final.

    Razonamiento de cada paso:
    - La transformacion lineal cambia la paleta manteniendo detalles,
      es más controlable que filtros no-lineales aleatorios.
    - El viñeteado (vignette) dirige la atención al centro y refuerza
      la sensación antigua/analógica.
    - Gamma corrige el brillo global para que el resultado final
      sea visualmente agradable.
    """
    # Aplicar la matriz de transformación de color; cv2.transform opera en float32
    f = cv2.transform(img.astype(np.float32), VINTAGE_MAT)

    # Crear máscara de viñeteado radial basada en la distancia al centro
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    # Normalizamos por el mayor de dimensiones para que el viñeteado
    # tenga una escala coherente en distintas resoluciones.
    vignette = 1.0 - 0.35 * (((x - cx)**2 + (y - cy)**2) / (max(h, w)**2))

    # Aplicar viñeteado al resultado de la transformación de color
    # vignette[..., None] expande a tres canales para multiplicar cada componente RGB.
    f *= vignette[..., None]

    # Volver a uint8 con clipping
    f = f.clip(0, 255).astype(np.uint8)

    # Pequeña corrección gamma final para ajustar luminosidad/contraste perceptual
    return apply_gamma(f, gamma=1.07)

if __name__ == "__main__":
    # Ruta de entrada: si se pasa argumento por línea de comandos lo usamos,
    # si no, se usa assets/foto.jpg por defecto (útil para pruebas).
    path_in = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/foto.jpg")
    img = cv2.imread(str(path_in))
    if img is None:
        raise SystemExit(f"No se pudo abrir {path_in}")

    # Aplicar filtro vintage y guardar el resultado.
    # Mantener nombres simples facilita pruebas manuales.
    out = vintage(img)
    cv2.imwrite("foto_vintage.jpg", out)
    print("Listo: foto_vintage.jpg")
