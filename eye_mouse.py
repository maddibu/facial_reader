import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Alias para simplificar el acceso a las clases de MediaPipe
BaseOptions = mp_python.BaseOptions
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Configuración del detector de rostros
# model_asset_path: ruta al archivo .task descargado (debe estar en la misma carpeta)
# running_mode IMAGE: procesa frame por frame (vs VIDEO o LIVE_STREAM que usan timestamps)
# num_faces: cuántos rostros detectar simultáneamente (1 es suficiente para eye tracking)
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=False,        # no necesitamos expresiones faciales
    output_facial_transformation_matrixes=False,  # no necesitamos transformaciones 3D
)

# 0 = cámara por defecto
# 1 = webcam externa
# 3 = OBS virtual camera
cap = cv2.VideoCapture(3)

# Configura la resolución y la tasa de cuadros por segundo (FPS)
# 640x480 es un buen balance: suficiente detalle para el iris, sin sobrecargar la CPU
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# "with" garantiza que el landmarker se cierra correctamente al salir,
# liberando memoria y recursos del modelo
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        # cap.read() devuelve dos valores:
        # ret  → True si el frame llegó bien, False si la cámara falló o se desconectó
        # frame → la imagen capturada como array de numpy (altura x ancho x 3 canales BGR)
        ret, frame = cap.read()
        if not ret:
            break

        # Voltea la imagen horizontalmente (modo espejo)
        # Sin esto, mover los ojos a la derecha mueve el cursor a la izquierda
        frame = cv2.flip(frame, 1)

        # OpenCV captura en BGR (azul-verde-rojo)
        # MediaPipe espera RGB (rojo-verde-azul)
        # Si no conviertes, los colores están invertidos y la detección falla
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Envuelve el array de numpy en el formato que MediaPipe entiende
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ejecuta el modelo sobre el frame actual
        # detect() es la API nueva (MediaPipe 0.10+)
        # La API vieja usaba .process() — ya no funciona con esta versión
        results = landmarker.detect(mp_image)

        # face_landmarks es una lista de rostros detectados
        # cada rostro tiene 478 puntos: 468 del rostro + 10 del iris (468-477)
        # si no hay rostro en cámara, la lista está vacía
        if results.face_landmarks:
            # aquí irá la lógica del iris en el siguiente paso
            pass

        # Muestra el frame en pantalla (usa BGR, por eso mostramos frame y no rgb_frame)
        cv2.imshow("camara", frame)

        # waitKey(1) espera 1ms entre frames — necesario para que la ventana responda
        # 0xFF es una máscara para compatibilidad con sistemas de 64 bits
        # ord('q') convierte el carácter 'q' a su valor ASCII (113)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera la cámara para que otros programas puedan usarla
cap.release()
# Cierra todas las ventanas abiertas por OpenCV
cv2.destroyAllWindows()