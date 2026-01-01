import threading
import time
from olympe.messages.ardrone3.Piloting import PCMD # type: ignore
from config import ORBIT_DT

orbit_active = False
orbit_thread = None


def orbit_loop(drone, clockwise=True):
    global orbit_active
    direction = 1 if clockwise else -1

    BASE_ROLL = 18
    BASE_YAW = 45

    while orbit_active:
        drone(PCMD(1, direction * BASE_ROLL, 0, direction * BASE_YAW, 0, 0))
        time.sleep(ORBIT_DT)

    drone(PCMD(0, 0, 0, 0, 0, 0))


def start_orbit(drone, clockwise=True):
    global orbit_active, orbit_thread

    if orbit_active:
        return

    orbit_active = True
    orbit_thread = threading.Thread(
        target=orbit_loop,
        args=(drone, clockwise),
        daemon=True
    )
    orbit_thread.start()


def stop_orbit(drone):
    global orbit_active
    orbit_active = False
    drone(PCMD(0, 0, 0, 0, 0, 0))
