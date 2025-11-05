import cv2
from vintage_filter import vintage

def main():
	"""
	Script simple de cámara que aplica el filtro 'vintage' frame a frame.

	Por qué así:
	- cv2.VideoCapture(0): usa la cámara por defecto. Cambiar a 1,2,... si hay varias.
	- Procesamos frame a frame en CPU usando la función vintage(img) del módulo vintage_filter.
	- waitKey(1) permite que OpenCV procese eventos de ventana y ofrece una forma no-bloqueante
	  de leer teclas; comprobamos ESC (27) para salir.
	- Siempre liberamos recursos (cap.release(), cv2.destroyAllWindows()) para evitar que la
	  cámara quede bloqueada tras cerrar el programa.
	"""
	cap = cv2.VideoCapture(0)  # cambia a 1/2 si tenés varias cámaras
	if not cap.isOpened():
		# Si no se pudo abrir la cámara abortamos con mensaje claro.
		raise SystemExit("No se pudo abrir la cámara")

	while True:
		# Leer un frame desde la cámara. 'ok' indica éxito; 'frame' es la imagen BGR.
		ok, frame = cap.read()
		if not ok:
			# Si falló la captura (cámara desconectada, error), salimos del bucle.
			break

		# Aplicar el filtro vintage definido en vintage_filter.py.
		# Reasoning: hacemos la conversión por frame para ver el resultado en tiempo real.
		out = vintage(frame)

		# Mostrar el resultado en una ventana. El título recuerda cómo salir (ESC).
		cv2.imshow("Vintage live (ESC para salir)", out)

		# waitKey(1) espera 1 ms y permite actualizar la ventana.
		# Comprobamos ESC (27) para salir. Usamos máscara & 0xFF por compatibilidad.
		if cv2.waitKey(1) & 0xFF == 27:
			# Salida limpia cuando el usuario presiona ESC.
			break

	# Liberar la cámara y cerrar ventanas OpenCV.
	# Muy importante para que otros procesos/OS puedan volver a usar la cámara.
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
