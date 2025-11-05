import cv2
import numpy as np
from pathlib import Path
import sys

# Colores aproximados del look "Rio": púrpura → magenta → naranja
PURPLE = np.array([180, 60, 170], np.float32)   # BGR
MAGENTA = np.array([190, 70, 190], np.float32)
ORANGE = np.array([40, 140, 250], np.float32)

def make_rio_gradient(h, w):
    """Gradiente diagonal tipo Instagram: púrpura (arriba-izq) → magenta (centro) → naranja (abajo-der)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    # normalizamos a [0,1] y generamos mezcla diagonal
    t = ((x + y) / (w + h)).reshape(h, w, 1)        # 0 en (0,0) → 1 en (w,h)
    mid = 0.45
    a = np.clip((t / mid), 0, 1)                    # 0..mid: PURPLE→MAGENTA
    b = np.clip(((t - mid) / (1 - mid)), 0, 1)      # mid..1:  MAGENTA→ORANGE
    grad = (1 - a) * PURPLE + a * MAGENTA
    grad = (1 - b) * grad  + b * ORANGE
    return grad.astype(np.uint8)

def softlight(base, overlay):
    """Mezcla tipo Soft Light (aprox). base y overlay en uint8 BGR."""
    B = base.astype(np.float32) / 255.0
    O = overlay.astype(np.float32) / 255.0
    out = (1 - 2*O) * (B**2) + 2*O*B
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def rio_filter(img_bgr, strength=0.65, vignette=True):
    h, w = img_bgr.shape[:2]
    grad = make_rio_gradient(h, w)

    # mezcla softlight controlada por 'strength'
    soft = softlight(img_bgr, grad)
    out = cv2.addWeighted(img_bgr, 1 - strength, soft, strength, 0)

    if vignette:
        # viñeta suave para cerrar bordes
        y, x = np.ogrid[:h, :w]
        cy, cx = h/2, w/2
        r2 = ((x - cx)**2 + (y - cy)**2) / (max(h, w)**2)
        mask = (1.0 - 0.5 * r2).clip(0.5, 1.0).astype(np.float32)
        out = (out.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)
    return out

if __name__ == "__main__":
    path_in = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/gatotriste.jpg")
    img = cv2.imread(str(path_in))
    if img is None:
        raise SystemExit(f"No se pudo abrir {path_in}")
    out = rio_filter(img, strength=0.75, vignette=True)
    cv2.imwrite("foto_rio.jpg", out)
    print("Listo: foto_rio.jpg")
