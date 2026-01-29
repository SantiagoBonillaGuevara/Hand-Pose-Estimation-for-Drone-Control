import cv2
from config import (
    CAMERA_INDEX,
    MAX_FRAMES,
    FEAT_DIM,
    MODEL_PATH,
    HAND_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    GESTURE_LABEL_TO_ID,
    WARNING_DURATION
)
from vision.hand_tracker import HandTracker
from features.extractor import extract_features
from model.classifier import GestureClassifier
from utils.buffer import SequenceBuffer
import state
import time

# Intentar importar la cola del dron (modo standalone si no existe)
try:
    from state import gesture_queue
except ImportError:
    gesture_queue = None


def run_camera():
    warning_msg = None
    warning_until = 0

    cap = cv2.VideoCapture(CAMERA_INDEX)
    buffer = SequenceBuffer(MAX_FRAMES)
    classifier = GestureClassifier(MODEL_PATH)

    last_gesture_name = "..."
    last_gesture_id = None

    print("ENTER: to send gesture | Q: exit")

    with HandTracker(HAND_MODEL_PATH) as tracker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, result = tracker.detect(frame)

            # =====================
            # Extraer landmarks
            # =====================
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                frame_feat = [v for lm in hand for v in (lm.x, lm.y, lm.z)]

                if len(frame_feat) == FEAT_DIM:
                    buffer.add(frame_feat)

            # =====================
            # Predicción
            # =====================
            if buffer.is_full():
                X_seq = buffer.to_array()[None, :, :]
                X_feat = extract_features(X_seq)

                gesture_name, conf = classifier.predict(
                    X_feat, CONFIDENCE_THRESHOLD
                )
                if gesture_name == "point" and result.hand_landmarks:
                    direction = point_direction(hand)
                    gesture_name = direction

                if gesture_name in GESTURE_LABEL_TO_ID:
                    last_gesture_name = gesture_name
                    last_gesture_id = GESTURE_LABEL_TO_ID[gesture_name]
                else:
                    last_gesture_name = gesture_name
                    last_gesture_id = None

            # =====================
            # Mostrar info
            # =====================
            cv2.putText(
                frame,
                f"Gesture: {last_gesture_name}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Q: exit",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if warning_msg and time.time() < warning_until:
                cv2.putText(
                    frame,
                    warning_msg,
                    (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),  # rojo
                    2
                )

            cv2.imshow("Hand Gesture Recognition", frame)
            # =====================
            # Teclado
            # =====================
            key = cv2.waitKey(1) & 0xFF

            # ENTER → enviar gesto
            if key in (10, 13):
                if last_gesture_id is not None:
                    if gesture_queue is not None:
                        gesture_queue.put(last_gesture_id)

                    print(
                        f"[SENT] Gesture: {last_gesture_name} | "
                        f"ID: {last_gesture_id}"
                    )

            # Q → salir
            elif key == ord('q'):
                if state.is_flying and gesture_queue is not None:
                    warning_msg = "Land first to exit"
                    warning_until = time.time() + WARNING_DURATION
                else:
                    if gesture_queue is not None:
                        gesture_queue.put(10)
                    break

    cap.release()
    cv2.destroyAllWindows()

def point_direction(hand_landmarks):
    """
    Retorna: 'forward', 'right', 'back', 'left'
    hand_landmarks: List[NormalizedLandmark]
    """

    wrist = hand_landmarks[0]      # WRIST
    index_tip = hand_landmarks[8]  # INDEX_FINGER_TIP

    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "forward" if dy < 0 else "back"


if __name__ == "__main__":
    run_camera()
