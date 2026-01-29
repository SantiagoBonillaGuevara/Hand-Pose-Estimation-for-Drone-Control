from enum import Enum
import threading
from threading import Event
from queue import Queue

running_event = threading.Event()
running_event.set()

gesture_queue = Queue()

class DroneState(Enum):
    IDLE = 0
    HOVER = 1
    MOVE_FORWARD = 2
    MOVE_BACK = 3
    MOVE_LEFT = 4
    MOVE_RIGHT = 5
    YAW_LEFT = 6
    YAW_RIGHT = 7
    ORBIT_CW = 8
    ORBIT_CCW = 9
    TAKEOFF = 10
    LANDING = 11

running_event = Event()
running_event.set()

current_state = DroneState.IDLE
state_lock = threading.Lock()

is_flying = False
current_altitude = 0.0
input_locked = False
running = True

def set_state(new_state):
    global current_state, state_lock
    with state_lock: 
        print("old state ",current_state," -> new state ",new_state)
        current_state = new_state