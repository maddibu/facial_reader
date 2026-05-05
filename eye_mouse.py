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
            iris = face[473]
            canto_ext = face[263]
            canto_int = face[362]
            sup = face[386]
            inf = face[374]


            # OpenCV's frame.shape[:2] returns (height, width)
            # defines x y y
            height, width = frame.shape[:2]
            iris_x = int(iris.x * width)
            iris_y = int(iris.y * height)
            canto_ext_x = int(canto_ext.x * width)
            canto_ext_y = int(canto_ext.y * height)
            canto_int_x = int(canto_int.x * width)
            canto_int_y = int(canto_int.y * height)
            sup_x = int(sup.x * width)
            sup_y = int(sup.y * height)
            inf_x = int(inf.x * width)
            inf_y = int(inf.y * height)
            promedio_x = int(((canto_ext_x + canto_int_x + sup_x + inf_x) / 4))
            promedio_y = int(((canto_ext_y + canto_int_y + sup_y + inf_y) / 4))


            #coordenadas de camara a coordenadas de pantalla
            screen_x = int(np.interp(iris_x, [0, width], [0, screen_w]))
            screen_y = int(np.interp(iris_y, [0, height], [0, screen_h]))
            canto_ext_screen_x = int(np.interp(canto_ext_x, [0, width], [0, screen_w]))
            canto_ext_screen_y = int(np.interp(canto_ext_y, [0, height], [0, screen_h]))
            canto_int_screen_x = int(np.interp(canto_int_x, [0, width], [0, screen_w]))
            canto_int_screen_y = int(np.interp(canto_int_y, [0, height], [0, screen_h]))
            sup_screen_x = int(np.interp(sup_x, [0, width], [0, screen_w]))
            sup_screen_y = int(np.interp(sup_y, [0, height], [0, screen_h]))
            inf_screen_x = int(np.interp(inf_x, [0, width], [0, screen_w]))
            inf_screen_y = int(np.interp(inf_y, [0, height], [0, screen_h]))



            promedio_screen_x = int(np.interp(promedio_x, [0, width], [0, screen_w]))
            promedio_screen_y = int(np.interp(promedio_y, [0, height], [0, screen_h]))

            smooth_x = smooth_x + SMOOTH * (screen_x - smooth_x)
            smooth_y = smooth_y + SMOOTH * (screen_y - smooth_y)

            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            #dibujo de landmark
            cv2.circle(frame, (iris_x, iris_y), 5, (0, 255, 0), -1)  # dibuja un círculo verde en el iris
            cv2.circle(frame, (canto_ext_x, canto_ext_y), 5, (0, 0, 255), -1)  # dibuja un círculo rojo en el canto externo
            cv2.circle(frame, (canto_int_x, canto_int_y), 5, (0, 0, 255), -1)  # dibuja un círculo azul en el canto interno
            cv2.circle(frame, (sup_x, sup_y), 5, (0, 0, 255), -1)  # dibuja un círculo cyan en la parte superior
            cv2.circle(frame, (inf_x, inf_y), 5, (0, 0, 255), -1)  # dibuja un círculo magenta en la parte inferior

            #texto coordenadas
            cv2.putText(frame, f'Iris_derecho: ({iris_x}, {iris_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Canto_ext: ({canto_ext_x}, {canto_ext_y})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Canto_int: ({canto_int_x}, {canto_int_y})', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Sup: ({sup_x}, {sup_y})', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Inf: ({inf_x}, {inf_y})', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'promedio_contorno: ({promedio_screen_x}, {promedio_screen_y})', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Muestra el frame en pantalla (usa BGR, por eso mostramos frame y no rgb_frame)
        cv2.imshow("camara", frame)

        # q para exit ventana
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 