from turtle import color
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import enum
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# Alias para simplificar el acceso a las clases de MediaPipe
BaseOptions = mp_python.BaseOptions
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode


########### Configuración del detector de rostros #############
    # model_asset_path: ruta al archivo .task descargado (debe estar en la misma carpeta)
    # running_mode IMAGE: procesa frame por frame (vs VIDEO o LIVE_STREAM que usan timestamps)
    # num_faces: cuántos rostros detectar simultáneamente
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,        # expresiones faciales
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


# valor inicial para evitar crash en el primer frame sin rostro
ROI_W  = 200
ROI_H  = 100
ESCALA = 4
roi_grande = np.zeros((ROI_H * ESCALA, ROI_W * ESCALA), dtype=np.uint8)


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
            
            MARGEN = 15
            ######################### OJO DERECHO #########################
            x1 = max(canto_int_x - MARGEN, 0)
            x2 = min(canto_ext_x + MARGEN, width)
            y1 = max(sup_y - MARGEN, 0)
            y2 = min(inf_y + MARGEN, height)

            roi_ojo = frame[y1:y2, x1:x2]
            gris = cv2.cvtColor(roi_ojo, cv2.COLOR_BGR2GRAY)
            contraste = cv2.convertScaleAbs(gris, alpha=2.0, beta=-30)
            roi_fijo   = cv2.resize(contraste, (ROI_W, ROI_H), interpolation=cv2.INTER_LINEAR)
            ####### ventana a visualizar #######
            roi_grande = cv2.resize(roi_fijo,  (ROI_W * ESCALA, ROI_H * ESCALA), interpolation=cv2.INTER_LINEAR)


            ################## DETECCIÓN DE PARPADEO #################
            distancia = inf_y - sup_y
            parpadeo = distancia < 20  # umbral de parpadeo, ajustar según sea necesario
            if parpadeo:
                cv2.putText(roi_grande, 'PARPADEO!', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            
            ################## Detenccion sonrisa derecha #################
            def detect_smile_right(blendshapes, roi_grande, threshold=0.5):

                bs = {b.category_name: b.score for b in blendshapes}
                
                smile_right = bs.get('mouthSmileRight', 0)
                smile_left  = bs.get('mouthSmileLeft', 0)
                mouth_close = bs.get('mouthClose', 0)

                # Sonrisa derecha dominante y boca no cerrada
                is_smiling_right = (
                    smile_right > threshold and
                    smile_right > smile_left and
                    mouth_close < 0.3
                )

                if is_smiling_right:
                    cv2.putText(roi_grande, 'DERECHA', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return is_smiling_right
            
            ################## Detenccion sonrisa izquierda #################
            def detect_smile_left(blendshapes, roi_grande, threshold=0.5):

                bs = {b.category_name: b.score for b in blendshapes}
                
                smile_right = bs.get('mouthSmileRight', 0)
                smile_left  = bs.get('mouthSmileLeft', 0)
                mouth_close = bs.get('mouthClose', 0)

                # Sonrisa izquierda dominante y boca no cerrada
                is_smiling_left = (
                    smile_left > threshold and
                    smile_left > smile_right and
                    mouth_close < 0.3
                )

                if is_smiling_left:
                    cv2.putText(roi_grande, 'IZQUIERDA', (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return is_smiling_left


            ################## Análisis de gestos #################
            if results.face_blendshapes:
                face_blendshapes = results.face_blendshapes[0]
                detect_smile_right(face_blendshapes, roi_grande)

            if results.face_blendshapes:
                face_blendshapes = results.face_blendshapes[0]
                detect_smile_left(face_blendshapes, roi_grande)

            ####### coordenadas ##########
            cv2.putText(roi_grande, f'Sup:               ({sup_x}, {sup_y})', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(roi_grande, f'Inf:               ({inf_x}, {inf_y})', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(roi_grande, f'Distancia:         {distancia}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ######### ventana (ojo derecho) #########
        cv2.imshow("ojo_derecho", roi_grande)
        key = cv2.waitKey(1) & 0xFF      
        if key == ord('q'):
            break    

cap.release()
cv2.destroyAllWindows() 