from state import DroneState
from motion.orbit import stop_orbit
from vision.snapshot import take_snapshot
import state as st

def handle_gesture(gesture_id, drone):

    if gesture_id == 4:
        #take_snapshot(drone)
        print("not available")
        return

    if gesture_id == 3:  # Open Palm
        st.input_locked = False
        stop_orbit(drone)
        st.set_state(DroneState.HOVER)
        return

    if gesture_id == 2:  # Closed Fist
        st.input_locked = True
        return

    if st.input_locked:
        print("input locked")
        return

    mapping = {
        7: DroneState.TAKEOFF,
        6: DroneState.LANDING,
        51: DroneState.MOVE_FORWARD,
        52: DroneState.MOVE_RIGHT,
        53: DroneState.MOVE_BACK,
        54: DroneState.MOVE_LEFT,
        8: DroneState.YAW_LEFT,
        9: DroneState.YAW_RIGHT,
        0: DroneState.ORBIT_CW,
        1: DroneState.ORBIT_CCW,
    }

    if gesture_id in mapping:
        st.set_state(mapping[gesture_id])
