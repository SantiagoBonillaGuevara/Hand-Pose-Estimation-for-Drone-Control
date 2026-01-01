import time
from olympe.messages.ardrone3.Piloting import PCMD  # type: ignore
import state as st
from state import DroneState
from motion.basic import handle_thumbs_up, handle_thumbs_down
from motion.orbit import start_orbit, stop_orbit
from config import CONTROL_LOOP_DT


def control_loop(drone):
    while st.running_event.is_set():

        with st.state_lock:
            state = st.current_state

        if state == DroneState.HOVER:
            print("stop")
            drone(PCMD(0, 0, 0, 0, 0, 0))

        elif state == DroneState.MOVE_FORWARD:
            stop_orbit(drone)
            drone(PCMD(1, 0, 40, 0, 0, 0))
        elif state == DroneState.MOVE_RIGHT:
            stop_orbit(drone)
            drone(PCMD(1, 40, 0, 0, 0, 0))
        elif state == DroneState.MOVE_BACK:
            stop_orbit(drone)
            drone(PCMD(1, 0, -40, 0, 0, 0))
        elif state == DroneState.MOVE_LEFT:
            stop_orbit(drone)
            drone(PCMD(1, -40, 0, 0, 0, 0))

        elif state == DroneState.YAW_LEFT:
            stop_orbit(drone)
            drone(PCMD(1, 0, 0, -40, 0, 0))

        elif state == DroneState.YAW_RIGHT:
            stop_orbit(drone)
            drone(PCMD(1, 0, 0, 40, 0, 0))

        elif state == DroneState.ORBIT_CW:
            start_orbit(drone, clockwise=True)
            st.set_state(DroneState.HOVER)

        elif state == DroneState.ORBIT_CCW:
            start_orbit(drone, clockwise=False)
            st.set_state(DroneState.HOVER)

        elif state == DroneState.TAKEOFF:
            stop_orbit(drone)
            handle_thumbs_up(drone)

        elif state == DroneState.LANDING:
            stop_orbit(drone)
            handle_thumbs_down(drone)
            if not st.is_flying:
                st.set_state(DroneState.IDLE)


        time.sleep(CONTROL_LOOP_DT)
