import cv2

def capture_image(camera_index=0):
    """Captures a single frame from the specified camera index."""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture image from webcam.")
        
    return frame

def show_camera_feed(camera_index=0):
    """Yields frames for a continuous camera feed until interrupted."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cap, frame
    finally:
        cap.release()
