import cv2
import numpy as np
import mediapipe as mp
from keras.layers import TFSMLayer

# Cargar el modelo exportado como capa de inferencia
modelo = TFSMLayer("modelo_pelicula3", call_endpoint="serving_default")

emociones = ['Enfado', 'Feliz', 'Miedo', 'Triste', 'Sorpresa', 'Neutral']

colores = {
    'Enfado': (0, 0, 255),           # Rojo
    'Feliz': (0, 215, 255),          # Amarillo
    'Miedo': (128, 0, 128),          # Morado
    'Triste': (180, 130, 70),        # Azul
    'Sorpresa': (0, 165, 255),       # Naranja
    'Neutral': (160, 160, 160)       # Gris
}

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe face detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = face_detection.process(rgb)

    if resultado.detections:
        for deteccion in resultado.detections:
            bbox = deteccion.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            x, y = max(x, 0), max(y, 0)
            w_box, h_box = min(w_box, frame.shape[1] - x), min(h_box, frame.shape[0] - y)

            rostro = frame[y:y + h_box, x:x + w_box]

            try:
                rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                rostro = cv2.resize(rostro, (96, 96))
                rostro = rostro.astype("float32") / 255.0
                rostro = np.expand_dims(rostro, axis=-1)
                rostro = np.expand_dims(rostro, axis=0)

                salida = modelo(rostro)
                clave_salida = list(salida.keys())[0]
                salida_np = salida[clave_salida].numpy()[0]

                idx = np.argmax(salida_np)
                emocion = emociones[idx]

                texto = f"{emocion}"
                color = colores.get(emocion, (255, 255, 255))

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.rectangle(frame, (x, y - 35), (x + w_box, y), color, -1)

                cv2.putText(frame, texto, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, texto, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 1, cv2.LINE_AA)

            except Exception as e:
                print("Error al procesar rostro:", e)

    cv2.imshow("Detección de Emociones", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()
