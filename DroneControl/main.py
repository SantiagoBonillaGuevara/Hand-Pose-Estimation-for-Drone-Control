import os
import logging
import sys
import olympe
import threading
from config import DRONE_IP
import state
from state import running_event
from control.fsm import control_loop
from control.gestures import handle_gesture

logging.disable(sys.maxsize)

# Forzar NetRaw discovery
os.environ["OLYMPE_NET_DISCOVERY"] = "netraw"

def main():
    drone = olympe.Drone(DRONE_IP)
    assert drone.connect()
    if drone.connect():
        print("Conectado al dron simulado")

    t = threading.Thread(target=control_loop, args=(drone,), daemon=True)
    t.start()

    try:
        while True:
            g = input("Gesto ID> ")
            if g == "q":
                break
            if g.isdigit():
                handle_gesture(int(g), drone)
                print("current state: ",state.current_state)
    finally:
        running_event.clear()
        drone.disconnect()


if __name__ == "__main__":
    main()
