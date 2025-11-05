import cv2
import mediapipe as mp
import numpy as np
import math

# ASSETS: rutas a imágenes PNG con canal alpha (RGBA). Usamos PNG con transparencia
# para poder mezclar accesorios (gafas, sombrero, bigote) sobre la imagen BGR de la cámara.
ASSETS = {
    "glasses": "assets/gafas.png",
    "hat": "assets/sombrero.png",
    "stache": "assets/bigote.png",
}

# Índices de puntos de MediaPipe Face Mesh usados por el script.
# Comentarios junto a cada constante para indicar qué parte del rostro representan.
ESQUINA_OJO_IZQ = 33      # Esquina externa del ojo izquierdo (útil para alinear gafas)
ESQUINA_OJO_DER = 263     # Esquina externa del ojo derecho
ESQUINA_BOCA_IZQ = 61     # Esquina izquierda de la boca (para ancho del bigote)
ESQUINA_BOCA_DER = 291    # Esquina derecha de la boca
LABIO_SUPERIOR = 13       # Punto central del labio superior (para posicionar el bigote)
FRENTE = 10               # Punto en la frente (para colocar el sombrero arriba)
OREJA_IZQ = 234           # Punto aproximado oreja izquierda (para ancho de cabeza)
OREJA_DER = 454           # Punto aproximado oreja derecha


def cargar_imagen(ruta):
    # Carga una imagen PNG con canal alpha (transparencia).
    # Por qué: necesitamos el canal alpha para mezclar (componer) el accesorio sobre el frame.
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        raise SystemExit(f"Error: {ruta} no encontrado o sin transparencia")
    return img


def calcular_angulo_y_distancia(punto1, punto2):
    # Calcula el ángulo (en grados) entre dos puntos y la distancia euclidiana.
    # Uso: el ángulo sirve para rotar accesorios (p. ej. gafas/sombrero) para que sigan la inclinación del rostro.
    dx = punto2[0] - punto1[0]
    dy = punto2[1] - punto1[1]
    angulo = math.degrees(math.atan2(dy, dx))
    distancia = math.hypot(dx, dy)
    return angulo, distancia


def superponer_imagen(fondo, accesorio, centro, escala=1.0, angulo=0.0):
    """
    Superpone una imagen RGBA (accesorio) sobre una imagen BGR (fondo).
    - Se redimensiona el accesorio según 'escala' y se rota según 'angulo'.
    - El accesorio se centra en 'centro' (x,y).
    - El canal alpha del PNG determina la mezcla (composición) sobre el fondo.
    Motivos/consideraciones:
    - Se usa borderMode reflect/constant al rotar/remap para evitar artefactos en bordes.
    - Se clippean coordenadas para no intentar escribir fuera del frame.
    """
    alto_fondo, ancho_fondo = fondo.shape[:2]
    alto_acc, ancho_acc = accesorio.shape[:2]

    # Redimensionar el accesorio según la escala (mantener mínimo 1 píxel)
    if escala != 1.0:
        nuevo_ancho = max(1, int(ancho_acc * escala))
        nuevo_alto = max(1, int(alto_acc * escala))
        accesorio = cv2.resize(accesorio, (nuevo_ancho, nuevo_alto))
        alto_acc, ancho_acc = accesorio.shape[:2]

    # Rotar el accesorio alrededor de su centro si se requiere.
    # Se usa BORDER_CONSTANT con valor (0,0,0,0) para mantener transparencia en zonas añadidas por la rotación.
    if angulo != 0.0:
        matriz = cv2.getRotationMatrix2D((ancho_acc / 2, alto_acc / 2), angulo, 1.0)
        accesorio = cv2.warpAffine(accesorio, matriz, (ancho_acc, alto_acc),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Calcular rectángulo donde se colocará el accesorio (centrado en 'centro')
    x_centro, y_centro = int(centro[0]), int(centro[1])
    x_inicio = x_centro - ancho_acc // 2
    y_inicio = y_centro - alto_acc // 2
    x_fin = x_inicio + ancho_acc
    y_fin = y_inicio + alto_acc

    # Si el accesorio queda completamente fuera del frame, no hacer nada.
    if x_fin <= 0 or y_fin <= 0 or x_inicio >= ancho_fondo or y_inicio >= alto_fondo:
        return fondo

    # Clippear las coordenadas para quedarse dentro del frame
    x_inicio_clip = max(0, x_inicio)
    y_inicio_clip = max(0, y_inicio)
    x_fin_clip = min(ancho_fondo, x_fin)
    y_fin_clip = min(alto_fondo, y_fin)

    # Extraer la región del accesorio que entra en el frame
    accesorio_recortado = accesorio[
        y_inicio_clip - y_inicio:y_fin_clip - y_inicio,
        x_inicio_clip - x_inicio:x_fin_clip - x_inicio
    ]

    if accesorio_recortado.size == 0:
        return fondo

    # Separar canales BGR y alpha. Convertir a float para mezcla lineal.
    bgr = accesorio_recortado[:, :, :3].astype(np.float32)
    alpha = accesorio_recortado[:, :, 3:].astype(np.float32) / 255.0

    # Extraer ROI del fondo y mezclar: out = alpha * accesorio + (1-alpha) * fondo
    roi = fondo[y_inicio_clip:y_fin_clip, x_inicio_clip:x_fin_clip].astype(np.float32)
    resultado = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)
    fondo[y_inicio_clip:y_fin_clip, x_inicio_clip:x_fin_clip] = resultado

    return fondo


def main():
    # Cargar las imágenes de accesorios en memoria (una vez al inicio)
    # Ventaja: evita volver a leer archivos PNG cada frame (mejor rendimiento).
    imagenes = {nombre: cargar_imagen(ruta) for nombre, ruta in ASSETS.items()}

    # Inicializar MediaPipe Face Mesh:
    # - max_num_faces=1 porque la demo aplica accesorios a una sola persona.
    # - las confianzas controlan sensibilidad/detección vs ruido.
    detector_rostro = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Abrir la cámara por defecto (0). Si falla, se aborta con mensaje claro.
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    print("Presiona ESC para salir")

    while True:
        # Capturar frame en BGR (OpenCV). Si falla, salimos del bucle.
        ok, frame = camara.read()
        if not ok:
            break

        alto, ancho = frame.shape[:2]

        # MediaPipe trabaja con RGB, por eso convertimos antes de procesar.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar para obtener landmarks (puntos del rostro)
        resultados = detector_rostro.process(frame_rgb)

        # Si detectó por lo menos un rostro, calculamos posiciones y superponemos accesorios
        if resultados.multi_face_landmarks:
            puntos = resultados.multi_face_landmarks[0].landmark

            # Función auxiliar: convierte coordenadas normalizadas (0..1) a píxeles.
            def obtener_punto(indice):
                return (int(puntos[indice].x * ancho), int(puntos[indice].y * alto))

            # ---------- GAFAS ----------
            # Usamos las esquinas externas de los ojos para:
            # - calcular el ángulo de rotación
            # - calcular el ancho aproximado que deben tener las gafas
            ojo_izq = obtener_punto(ESQUINA_OJO_IZQ)
            ojo_der = obtener_punto(ESQUINA_OJO_DER)
            angulo_gafas, ancho_ojos = calcular_angulo_y_distancia(ojo_izq, ojo_der)
            centro_ojos = ((ojo_izq[0] + ojo_der[0]) // 2, (ojo_izq[1] + ojo_der[1]) // 2)
            # Factor de escala 1.50 para que las gafas queden algo más anchas que la distancia entre esquinas.
            escala_gafas = (ancho_ojos / imagenes["glasses"].shape[1]) * 1.50
            frame = superponer_imagen(frame, imagenes["glasses"], centro_ojos,
                                       escala=escala_gafas, angulo=angulo_gafas)

            # ---------- BIGOTE ----------
            # Calculamos ancho de la boca y colocamos el bigote ligeramente por encima del labio superior.
            boca_izq = obtener_punto(ESQUINA_BOCA_IZQ)
            boca_der = obtener_punto(ESQUINA_BOCA_DER)
            labio_sup = obtener_punto(LABIO_SUPERIOR)
            angulo_bigote, ancho_boca = calcular_angulo_y_distancia(boca_izq, boca_der)
            centro_x_boca = (boca_izq[0] + boca_der[0]) // 2
            # Desplazamiento hacia arriba: se combina un porcentaje de la altura y del ancho de la boca
            desplazamiento = int(max(0.02 * alto, 0.08 * ancho_boca))
            centro_bigote = (centro_x_boca, labio_sup[1] - desplazamiento)
            escala_bigote = (ancho_boca / imagenes["stache"].shape[1]) * 1.1
            frame = superponer_imagen(frame, imagenes["stache"], centro_bigote,
                                       escala=escala_bigote, angulo=angulo_bigote)

            # ---------- SOMBRERO ----------
            # Usamos distancia oreja-oreja para estimar ancho de la cabeza y colocar el sombrero arriba.
            oreja_izq = obtener_punto(OREJA_IZQ)
            oreja_der = obtener_punto(OREJA_DER)
            frente = obtener_punto(FRENTE)
            angulo_sombrero, ancho_cabeza = calcular_angulo_y_distancia(oreja_izq, oreja_der)
            # Se coloca el centro del sombrero algo por encima del punto de la frente
            centro_sombrero = (frente[0], int(frente[1] - 0.25 * ancho_cabeza))
            escala_sombrero = (2.0 * ancho_cabeza) / imagenes["hat"].shape[1]
            frame = superponer_imagen(frame, imagenes["hat"], centro_sombrero,
                                       escala=escala_sombrero, angulo=angulo_sombrero)

        # Mostrar resultado y permitir salida con ESC
        cv2.imshow("Filtro de Accesorios - Presiona ESC para salir", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Liberar cámara y cerrar ventanas
    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()