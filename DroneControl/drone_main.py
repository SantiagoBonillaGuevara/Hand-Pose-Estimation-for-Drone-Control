import os
import logging
import sys
import olympe
import threading
from config import DRONE_IP
import state
from control.fsm import control_loop
from control.gestures import handle_gesture

logging.disable(sys.maxsize)
os.environ["OLYMPE_NET_DISCOVERY"] = "netraw"

def drone_loop():
    drone = olympe.Drone(DRONE_IP)
    assert drone.connect()
    if drone.connect():
        print("Conectado al dron simulado")

    # Inicia control loop en hilo separado
    t = threading.Thread(target=control_loop, args=(drone,), daemon=True)
    t.start()

    try:
        while state.running_event.is_set():
            try:
                # leer gesto de la cola
                gesture_id = state.gesture_queue.get(timeout=0.1)
                if gesture_id == 10:
                    break
                handle_gesture(gesture_id, drone)
            except:
                continue
    finally:
        state.running_event.clear()
        drone.disconnect()


if __name__ == "__main__":
    drone_thread = threading.Thread(target=drone_loop, daemon=True)
    drone_thread.start()

    # también correr cámara en otro hilo
    from camera import run_camera
    run_camera()

    drone_thread.join()
