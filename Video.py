import cv2
import csv
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import mediapipe as mp

video_path = "Videos/Peliculaaa.mp4"
csv_path = "emociones_video.csv"
output_path = "Videos/Video.Emociones.mp4"

emociones = ['Enfado', 'Feliz', 'Miedo', 'Triste', 'Sorpresa', 'Neutral']
colores = {
    'Enfado': (0, 0, 255),           # Rojo
    'Feliz': (0, 215, 255),          # Amarillo
    'Miedo': (128, 0, 128),          # Morado
    'Triste': (180, 130, 70),        # Azul
    'Sorpresa': (0, 165, 255),       # Naranja
    'Neutral': (160, 160, 160)       # Gris
}


modelo = TFSMLayer("modelo_pelicula3", call_endpoint="serving_default")
video = cv2.VideoCapture(video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# CSV
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Segundo", "Emocion", "Confianza"])

frame_num = 0
emocion_anterior = "Neutral"
confianza_anterior = 0.0
ultimo_cambio = -10

# Mediapipe
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

while True:
    ret, frame = video.read()
    if not ret:
        break

    segundo_actual = frame_num // fps
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = face_detection.process(rgb)

    if resultado.detections:
        deteccion = max(resultado.detections, key=lambda d: d.score[0])
        bbox = deteccion.location_data.relative_bounding_box
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, width - x), min(h, height - y)

        rostro = frame[y:y + h, x:x + w]
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
            confianza = float(salida_np[idx])

            if confianza >= 0.2 and (frame_num - ultimo_cambio > fps * 2):
                emocion_anterior = emocion
                confianza_anterior = confianza
                ultimo_cambio = frame_num

            if frame_num - ultimo_cambio > fps * 2:
                emocion_anterior = emocion
                confianza_anterior = confianza
                ultimo_cambio = frame_num

            color = colores.get(emocion_anterior, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(frame, emocion_anterior, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            csv_writer.writerow([segundo_actual, emocion_anterior, round(confianza_anterior, 3)])

        except Exception as e:
            print(f" Error en el frame {frame_num} con detección: {e}")

    else:
        print(f" No se detectaron rostros en el frame {frame_num}")

    out.write(frame)
    frame_num += 1

video.release()
out.release()
csv_file.close()
face_detection.close()
print(" Análisis completado con éxito.")
