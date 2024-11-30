import cv2

# Abrir la cámara
captura = cv2.VideoCapture(0)

# Leer un cuadro de la cámara
ret, frame = captura.read()

# Si la captura fue exitosa, guardar el cuadro
if ret:
    cv2.imwrite('C:/Users/VRNK1/Documents/python/Captura2.jpg', frame)
    print("Imagen capturada y guardada como 'Captura2.jpg'")
else:
    print("Error: No se pudo capturar la imagen.")

# Liberar la cámara
captura.release()
