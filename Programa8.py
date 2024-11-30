import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar las imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    # Procesa la imagen como una matriz
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Generación de datos de ejemplo (se necesitan imágenes etiquetadas)
def load_dataset(image_paths, labels, target_size=(64, 64)):
    images = [preprocess_image(img, target_size) for img in image_paths]
    return np.array(images), np.array(labels)

# Ejemplo de datos (modifica con tus propios datos)
image_paths = [
    'C:/Users/VRNK1/Documents/python/triangulo.jpg',
    'C:/Users/VRNK1/Documents/python/cuadrado.jpg',
    'C:/Users/VRNK1/Documents/python/circulo.jpg',
    # Agrega más imágenes aquí
]
labels = [0, 1, 2]  # Etiquetas: 0-Triángulo, 1-Cuadrado, 2-Círculo
X, y = load_dataset(image_paths, labels)
y = to_categorical(y, num_classes=3)  # Codificación one-hot

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation para mejorar el modelo
datagen = ImageDataGenerator(
    rotation_range=40,       # Rotar las imágenes aleatoriamente
    width_shift_range=0.2,   # Desplazar las imágenes horizontalmente
    height_shift_range=0.2,  # Desplazar las imágenes verticalmente
    shear_range=0.2,         # Aplicar un corte (shear)
    zoom_range=0.2,          # Zoom aleatorio
    horizontal_flip=True,    # Voltear las imágenes horizontalmente
    fill_mode='nearest'      # Método de relleno de los píxeles faltantes
)

# Ajustar los datos de entrenamiento con augmentation
datagen.fit(X_train)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(256, activation='relu'),  # Aumentar el número de neuronas
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: triángulo, cuadrado, círculo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con el data augmentation
model.fit(datagen.flow(X_train, y_train, batch_size=16), epochs=20, validation_data=(X_val, y_val))

# Guardar el modelo entrenado
model.save('shape_detector_model.h5')

# Usar el modelo en tu script original
def predict_shape(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)  # Ahora acepta matrices
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Cargar el modelo guardado
model = tf.keras.models.load_model('shape_detector_model.h5')

# Procesar la imagen para detección de contornos y usar el modelo
image = cv2.imread('C:/Users/VRNK1/Documents/python/FigurasColores1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 200)  # Ajuste de parámetros de Canny
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Obtener la ruta del directorio actual donde está el script
output_folder = os.getcwd()  # Carpeta donde está el código Python

for i, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    roi = image[y:y+h, x:x+w]  # Región de interés
    if roi.size > 0:
        class_idx = predict_shape(roi, model)
        label = ["Triangulo", "Cuadrado", "Circulo"][class_idx]
        cv2.putText(image, label, (x, y-5), 1, 1, (0, 255, 0), 1)
    
    # Dibujar los contornos
    cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

    # Guardar la imagen con el texto sobre la figura
    output_image_path = os.path.join(output_folder, f"detected_shape_{i+1}.png")
    cv2.imwrite(output_image_path, image)  # Guardar la imagen en la carpeta del código

# Mostrar la imagen final con etiquetas
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

