from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD # type: ignore
from config import MIN_ALTITUDE, MAX_ALTITUDE
import state as st

def hover(drone):
    drone(PCMD(0, 0, 0, 0, 0, 0))


def handle_thumbs_up(drone):
    if not st.is_flying:
        print("takeOff")
        drone(TakeOff()).wait()
        st.is_flying = True
        st.current_altitude = 1.0  # altitud de hover inicial
    else:
        # Subir gradualmente
        print("Gradually rise")
        if st.current_altitude < MAX_ALTITUDE:
            drone(PCMD(1, 0, 0, 0, 30, 0))  # gz positivo
            st.current_altitude += 0.1

def handle_thumbs_down(drone):
    if st.is_flying:
        if st.current_altitude > MIN_ALTITUDE:
            # Bajar gradualmente
            print("Gradually lower")
            drone(PCMD(1, 0, 0, 0, -30, 0))  # gz negativo
            st.current_altitude -= 0.1
        else:
            # Altitud mínima alcanzada → aterrizar
            print("land")
            drone(Landing()).wait()
            st.is_flying = False
            st.current_altitude = 0.0