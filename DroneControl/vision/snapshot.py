import cv2
from threading import Event

_snapshot_in_progress = Event()

def take_snapshot(drone, filename="snapshot.jpg"):

    if _snapshot_in_progress.is_set():
        print("Snapshot en curso, ignorado")
        return

    _snapshot_in_progress.set()

    def yuv_cb(yuv_frame):
        try:
            yuv = yuv_frame.as_ndarray()
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            cv2.imwrite(filename, bgr)
            print(f"Foto guardada: {filename}")
        finally:
            yuv_frame.unref()
            drone.stop_video_streaming()
            _snapshot_in_progress.clear()

    drone.start_video_streaming(yuv_frame_cb=yuv_cb)

