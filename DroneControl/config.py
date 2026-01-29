import os

GESTURE_LABEL_TO_ID = {
    "circleClock": 0,
    "circleCounterclock": 1,
    "closedFist": 2,            # LANDING
    "openPalm": 3,        # TAKEOFF
    "pinch": 4,
    "forward": 51,
    "right": 52,
    "back": 53,
    "left": 54,
    "thumbsDown": 6,
    "thumbsUp": 7,
    "waveLeft": 8,
    "waveRight": 9
}

DRONE_IP = "10.202.0.1"

MIN_ALTITUDE = 0.3
MAX_ALTITUDE = 10.0

CONTROL_LOOP_DT = 0.1
ORBIT_DT = 0.05

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "TrainedModels",
    "MediaPipe",
    "trained_model_mediapipe.pkl"
)
HAND_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "TrainedModels",
    "MediaPipe",
    "hand_landmarker.task"
)

MAX_FRAMES = 48
NUM_LANDMARKS = 21
FEAT_DIM = NUM_LANDMARKS * 3

CONFIDENCE_THRESHOLD = 0.3
CAMERA_INDEX = 0

WARNING_DURATION = 2.0