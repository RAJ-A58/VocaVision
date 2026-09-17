import cv2

def show_camera_feed(camera_index=0):
    """
    Generator that yields frames from the webcam until the feed fails.
    The VideoCapture object is fully encapsulated and released on exit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Check that your camera is connected and not in use by another app."
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame  # Only yield the frame — cap stays encapsulated
    finally:
        cap.release()
