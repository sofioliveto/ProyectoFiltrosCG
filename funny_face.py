import cv2
import numpy as np
from pathlib import Path

def sinus_warp(img, amp=12.0, period=18.0):
    """
    Aplica una deformación senoidal horizontal a la imagen.
    - amp: amplitud del desplazamiento horizontal en píxeles.
    - period: periodo vertical de la onda (cuánto separan las "ondas" en px).
    Razonamiento: crear un efecto de ondulación simple variando x en función de y.
    """
    h, w = img.shape[:2]
    # yy, xx son coordenadas enteras de píxeles; usamos float para los mapas de remapeo
    yy, xx = np.indices((h, w), dtype=np.float32)
    # mapx desplaza cada columna xx por una senoide dependiente de la fila yy.
    # mapy queda igual (no hay desplazamiento vertical).
    mapx = xx + amp * np.sin(yy / period)
    mapy = yy
    # remap aplica la transformación basada en mapx/mapy.
    # BORDER_REFLECT101 reduce artefactos en los bordes reflejando píxeles.
    return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def bulge_roi(img, roi, strength=0.55):
    """
    Aplica un efecto 'bulge' (hinchado/achicado radial) dentro de una región rectangular (ROI).
    - roi: tupla (x, y, w, h) que define el rectángulo a transformar.
    - strength: controla la intensidad del bulge; valores >0 producen un efecto
      más pronunciado (0 = identidad).
    Razonamiento:
    - Normalizamos coordenadas respecto al centro del ROI para trabajar en un sistema radial.
    - Calculamos radio r y aplicamos una función r' = r^(1 - strength) para comprimir/expandir.
    - Reconstruimos coordenadas (nx2, ny2) y usamos remap para obtener la imagen distorsionada.
    """
    x, y, w, h = roi
    sub = img[y:y+h, x:x+w].copy()
    hh, ww = sub.shape[:2]

    # Coordenadas dentro del sub-ROI
    yy, xx = np.indices((hh, ww), dtype=np.float32)

    # Normalizamos a [-1,1] en ambas direcciones con centro en (ww/2, hh/2)
    nx = (xx - ww/2) / (ww/2)
    ny = (yy - hh/2) / (hh/2)

    # r es la distancia radial normalizada; máscara para limitar la transformación al interior del círculo unitario
    r = np.sqrt(nx*nx + ny*ny)
    mask = r <= 1.0

    # r_prime modifica el flujo radial; r' = r^(1 - strength) produce el efecto de bulge
    # (cuando strength>0 se "aplana" r, acercando puntos hacia el centro o alejándolos)
    r_prime = r.copy()
    r_prime[mask] = r[mask] ** (1.0 - strength)

    # Mantenemos el ángulo theta y reconstruimos las coordenadas modificadas
    theta = np.arctan2(ny, nx)
    nx2 = r_prime * np.cos(theta)
    ny2 = r_prime * np.sin(theta)

    # Convertimos de vuelta a coordenadas de píxel en el sub-ROI
    mapx = (nx2 * (ww/2) + ww/2).astype(np.float32)
    mapy = (ny2 * (hh/2) + hh/2).astype(np.float32)

    # Remapeo del sub-ROI usando los mapas calculados
    distorted = cv2.remap(sub, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    # Insertamos el sub-ROI modificado de vuelta en una copia de la imagen original
    out = img.copy()
    out[y:y+h, x:x+w] = distorted
    return out

if __name__ == "__main__":
    # Ruta a la imagen de prueba; usar Path facilita compatibilidad y mensajes claros
    path = Path("assets/icardi.jpg")
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"No se pudo abrir {path}")

    # -------------------
    # Aplicar sinus_warp
    # - amp y period elegidos para un efecto visible pero no destructivo.
    # - Guardamos resultado en archivo para inspección.
    # Razonamiento: la onda se nota mejor con amp ~10-20 y period entre 10-30 para tamaños de imagen típicos.
    sin_out = sinus_warp(img, amp=12.0, period=18.0)
    cv2.imwrite("rostro_sinus.jpg", sin_out)

    # -------------------
    # Aplicar bulge_roi sobre una región aproximada del rostro
    # - Seleccionamos ROI relativo al tamaño de la imagen para que la demo funcione en distintas resoluciones.
    # - strength = 0.55 da un efecto apreciable sin romper la estructura facial.
    h, w = img.shape[:2]
    rx, ry = int(w*0.25), int(h*0.3)         # origen del ROI: centrado horizontalmente y algo hacia arriba
    rw, rh = int(w*0.5), int(h*0.25)        # tamaño del ROI: la mitad del ancho y cuarto de la altura
    # Razonamiento: estos valores buscan cubrir la región de la cara típica en un retrato.
    bulge_out = bulge_roi(img, (rx, ry, rw, rh), strength=0.55)
    cv2.imwrite("rostro_bulge.jpg", bulge_out)

    print("Listo: rostro_sinus.jpg y rostro_bulge.jpg")
