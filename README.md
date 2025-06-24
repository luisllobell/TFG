# TFG
Reconocimiento de emociones faciales con CNN

Este proyecto corresponde a mi Trabajo de Fin de Grado (TFG), cuyo objetivo principal ha sido desarrollar un sistema capaz de reconocer emociones faciales en imágenes y vídeos utilizando redes neuronales convolucionales (CNN). A lo largo del proyecto se ha realizado todo el ciclo completo: recopilación de datos, etiquetado, preprocesado, entrenamiento del modelo, pruebas en vídeo y puesta en producción.


Estructura del repositorio:

Entrenamiento.ipynb: Jupyter Notebook con todo el proceso de entrenamiento del modelo, desde la carga de datos hasta su evaluación.

webcam.py: Script en Python que permite detectar emociones en tiempo real usando la webcam.

Video.py: Script que analiza un vídeo previamente grabado, frame a frame, para detectar y etiquetar emociones.

Frames2.zip: Contiene los frames recortados (caras) del vídeo casero y del vídeo final. Estas imágenes han sido usadas para entrenar el modelo final, mejorando su precisión.

datos.tfg.zip: Contiene las imágenes originales del dataset FER-2013.

Depósito.pdf: Documento PDF oficial entregado a la universidad.

README.md: Este archivo.


Objetivos del proyecto:

Comprender y aplicar los fundamentos de las redes neuronales convolucionales (CNN).

Realizar un sistema de reconocimiento de emociones faciales capaz de operar tanto en tiempo real (webcam) como sobre vídeos pregrabados.

Documentar y visualizar los resultados del modelo, explicando las decisiones de diseño y arquitectura.

Aumentar la precisión del modelo original combinando datasets y aplicando data augmentation.


Tecnologías utilizadas:

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Jupyter Notebook

PyCharm

OBS Studio / iMovie (para grabación y edición de los vídeos)

GitHub (entorno de entrega y control de versiones)


Dataset:

Se han utilizado principalmente dos fuentes de datos:

FER-2013: Dataset estándar de reconocimiento de emociones faciales con 7 emociones (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

Frames2.zip: Imágenes extraídas de vídeos reales (video casero y vídeo final). Estas imágenes se han recortado automáticamente para quedarse solo con el rostro usando cascadas Haar de OpenCV.

Las imágenes han sido organizadas en carpetas por emoción (con nombres en español) para facilitar el entrenamiento.


Aplicaciones desarrolladas:

webcam.py: Permite ver en tiempo real la emoción detectada sobre el rostro capturado por la cámara, mostrando además el nombre de la emoción en español y un color distinto por emoción.

Video.py: Lee un vídeo completo, analiza cada frame y genera un nuevo vídeo etiquetado, además de un archivo CSV con la emoción detectada por segundo.


Lecciones aprendidas:

Durante el desarrollo de este proyecto he aprendido a trabajar con modelos de redes neuronales convolucionales ya entrenados, comprendiendo su arquitectura y comportamiento, y aplicando mejoras significativas mediante la ampliación del dataset, técnicas de aumento de datos (data augmentation) y ajustes de entrenamiento. He integrado datos propios (extraídos de vídeos reales) para reforzar clases minoritarias y aumentar la precisión del modelo en condiciones reales. También he trabajado en el etiquetado y preprocesado de imágenes, la automatización del flujo de detección y recorte facial con OpenCV, y la implementación de scripts funcionales para analizar vídeo en tiempo real o vídeo grabado. Además, he consolidado mis habilidades en herramientas como Jupyter, PyCharm, OBS, iMovie y GitHub, y he ganado experiencia en el desarrollo completo de una solución basada en IA, desde la preparación del dataset hasta su puesta en producción.


Cómo ejecutar el proyecto:

Clona este repositorio.

Asegúrate de tener Python 3.9+ y las librerías necesarias (tensorflow, opencv-python, numpy, etc.).

Descomprime los archivos Frames2.zip y datos.tfg.zip en la raíz del proyecto.

Ejecuta Entrenamiento.ipynb para ver el proceso de entrenamiento.

Ejecuta webcam.py o Video.py para probar la detección de emociones.





