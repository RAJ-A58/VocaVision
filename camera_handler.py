import cv2

def capture_frame(camera_index=0, warmup_frames=4):
    """
    Captures a single frame from the webcam, warming up exposure first.
    Releases the camera immediately after capture.
    Returns: frame (np.ndarray) or raises RuntimeError.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Check that your camera is connected and not in use by another app."
        )
    try:
        # Warmup sensor for exposure and white balance
        for _ in range(warmup_frames):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to grab frame from webcam at index {camera_index}.")
        return frame
    finally:
        cap.release()

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
            yield frame
    finally:
        cap.release()
