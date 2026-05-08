from turtle import color

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Alias para simplificar el acceso a las clases de MediaPipe
BaseOptions = mp_python.BaseOptions
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode





# Configuración del detector de rostros
# model_asset_path: ruta al archivo .task descargado (debe estar en la misma carpeta)
# running_mode IMAGE: procesa frame por frame (vs VIDEO o LIVE_STREAM que usan timestamps)
# num_faces: cuántos rostros detectar simultáneamente
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=False,        # no necesitamos expresiones faciales AUN
    output_facial_transformation_matrixes=False,  # no necesitamos transformaciones 3D
)




# 0 = cámara por defecto
# 1 = webcam externa (camo)
# 3 = OBS virtual camera
cap = cv2.VideoCapture(1)

# Configura la resolución y la tasa de cuadros por segundo (FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)





screen_w, screen_h = pyautogui.size()

smooth_x, smooth_y = screen_w // 2, screen_h // 2
SMOOTH = 0.15




PASOS_CALIBRACION = ["sup_izq", "sup_der", "centro", "inf_izq", "inf_der"]
paso_actual = 0
calibrado = False
calibracion = {}









# carga el modelo y entra en un loop continuo leyendo un frame de la cámara
# por iteración hasta que falle la lectura o el usuario salga
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltea la imagen horizontalmente (espejo)
        frame = cv2.flip(frame, 1)

        # OpenCV captura en BGR
        # MediaPipe espera RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Envuelve el array de numpy en el formato que MediaPipe entiende
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ejecuta el modelo sobre el frame actual
        results = landmarker.detect(mp_image)





        # face_landmarks es una lista de rostros detectados
        # cada rostro tiene 478 puntos: 468 del rostro + 10 del iris (468-477)
        # si no hay rostro en cámara, la lista está vacía
        if results.face_landmarks:
            face = results.face_landmarks[0]

            iris      = face[473]
            canto_ext = face[263]
            canto_int = face[362]
            sup       = face[386]
            inf       = face[374]


            # OpenCV's frame.shape[:2] returns (height, width)
            # defines x y y
            height, width = frame.shape[:2]

            iris_x      = int(iris.x * width)
            iris_y      = int(iris.y * height)
            canto_ext_x = int(canto_ext.x * width)
            canto_ext_y = int(canto_ext.y * height)
            canto_int_x = int(canto_int.x * width)
            canto_int_y = int(canto_int.y * height)
            sup_x       = int(sup.x * width)
            sup_y       = int(sup.y * height)
            inf_x       = int(inf.x * width)
            inf_y       = int(inf.y * height)
            promedio_x  = int(((canto_ext_x + canto_int_x + sup_x + inf_x) / 4))
            promedio_y  = int(((canto_ext_y + canto_int_y + sup_y + inf_y) / 4))





            distancia = inf_y - sup_y
            parpadeo_detectado = distancia < 13

            if parpadeo_detectado:
                cv2.putText(frame, f'CLICK', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            



            
            MARGEN = 15

            x1 = max(canto_int_x - MARGEN, 0)
            x2 = min(canto_ext_x + MARGEN, width)
            y1 = max(sup_y - MARGEN, 0)
            y2 = min(inf_y + MARGEN, height)

            roi_ojo = frame[y1:y2, x1:x2]
            gris = cv2.cvtColor(roi_ojo, cv2.COLOR_BGR2GRAY)
            contraste = cv2.convertScaleAbs(gris, alpha=2.0, beta=-30)

            roi_grande = cv2.resize(contraste, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

            cv2.imshow("ojo_derecho", roi_grande)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break





            #coordenadas de camara a coordenadas de pantalla
            screen_x = int(np.interp(iris_x, [0, width], [0, screen_w]))
            screen_y = int(np.interp(iris_y, [0, height], [0, screen_h]))



            promedio_screen_x = int(np.interp(promedio_x, [0, width], [0, screen_w]))
            promedio_screen_y = int(np.interp(promedio_y, [0, height], [0, screen_h]))

            smooth_x = smooth_x + SMOOTH * (screen_x - smooth_x)
            smooth_y = smooth_y + SMOOTH * (screen_y - smooth_y)

            pyautogui.moveTo(int(smooth_x), int(smooth_y))



            if not calibrado:
                punto_calibracion = PASOS_CALIBRACION[paso_actual]

                mensajes = {
                    "sup_izq": "Mira al punto superior izquierdo y parpadea",
                    "sup_der": "Mira al punto superior derecho y parpadea",
                    "centro": "Mira al centro y parpadea",
                    "inf_izq": "Mira al punto inferior izquierdo y parpadea",
                    "inf_der": "Mira al punto inferior derecho y parpadea",
                }
                cv2.putText(frame, mensajes[punto_calibracion], (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if parpadeo_detectado:
                    calibracion[punto_calibracion] = (iris_x, iris_y)
                    paso_actual += 1
                    
                    if paso_actual >= len(PASOS_CALIBRACION):
                        calibrado = True
                        print("Calibración completa:", calibracion)





            #dibujo de landmark
            cv2.circle(frame, (iris_x, iris_y), 5, (0, 255, 0), -1)  # dibuja un círculo verde en el iris
            cv2.circle(frame, (canto_ext_x, canto_ext_y), 5, (0, 0, 255), -1)  # dibuja un círculo rojo en el canto externo
            cv2.circle(frame, (canto_int_x, canto_int_y), 5, (0, 0, 255), -1)  
            cv2.circle(frame, (sup_x, sup_y), 5, (0, 0, 255), -1)  
            cv2.circle(frame, (inf_x, inf_y), 5, (0, 0, 255), -1)  

            #texto coordenadas
            cv2.putText(frame, f'Iris_derecho:      ({iris_x}, {iris_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Canto_ext:         ({canto_ext_x}, {canto_ext_y})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Canto_int:         ({canto_int_x}, {canto_int_y})', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Sup:               ({sup_x}, {sup_y})', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Inf:               ({inf_x}, {inf_y})', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'promedio_contorno: ({promedio_screen_x}, {promedio_screen_y})', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Distancia ojo: {distancia}px', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Muestra el frame en pantalla (usa BGR, por eso mostramos frame y no rgb_frame)
        cv2.imshow("camara", frame)

        # q para exit ventana
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 