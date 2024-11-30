import cv2
import numpy as np

# Ajustar el rango de color verde
verde_bajo = np.array([35, 50, 50], np.uint8)  # Rango mínimo de verde
verde_alto = np.array([85, 255, 255], np.uint8)  # Rango máximo de verde

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a HSV
    imagen_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear una máscara para los colores verdes
    mascara_verde = cv2.inRange(imagen_hsv, verde_bajo, verde_alto)

    # Aplicar un filtro de suavizado para mejorar la detección de círculos
    mascara_verde = cv2.GaussianBlur(mascara_verde, (15, 15), 0)

    # Mostrar la máscara verde
    cv2.imshow('Máscara Verde', mascara_verde)

    # Encontrar los círculos verdes utilizando HoughCircles
    circulos = cv2.HoughCircles(
        mascara_verde,          # La máscara verde para detectar los círculos
        cv2.HOUGH_GRADIENT,     # Método de detección de círculos
        dp=1.2,                 # Resolución de la acumulación
        minDist=30,             # Mínima distancia entre los centros de los círculos
        param1=50,              # Umbral de borde en la detección de bordes
        param2=30,              # Umbral de precisión para la detección de círculos
        minRadius=10,           # Radio mínimo de los círculos
        maxRadius=50            # Radio máximo de los círculos
    )

    # Si se encontraron círculos
    if circulos is not None:
        circulos = np.uint16(np.around(circulos[0, :]))  # Redondear a enteros
        for i, (x, y, r) in enumerate(circulos):
            # Dibujar el círculo verde
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            # Etiquetar el círculo con su número y el radio
            cv2.putText(frame, f"Círculo {i+1} - Radio: {r} px", (x - 50, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostrar la imagen original con los círculos detectados y las mediciones
    cv2.imshow('Círculos Verdes Detectados', frame)

    # Detener el programa cuando presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Guardar la imagen procesada como un archivo (puedes cambiar el nombre si lo prefieres)
    if cv2.waitKey(1) & 0xFF == ord('s'):  # Si presionas 's', guarda la imagen
        cv2.imwrite('circulos_detectados.png', frame)
        print("Imagen guardada como 'circulos_detectados.png'.")

cap.release()
cv2.destroyAllWindows()
