import threading
import pyttsx3

def initialize_tts():
    """
    Initializes the pyttsx3 Text-to-Speech engine.
    Raises a RuntimeError with a clear message if TTS is unavailable on this system.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        return engine
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Text-to-Speech engine: {e}\n"
            "On Windows, ensure the SAPI5 voice driver is available. "
            "On Linux, ensure 'espeak' is installed."
        ) from e

def speak(engine, text):
    """
    Speaks the given text synchronously (blocks until speech completes).
    Use speak_async() from the main loop to avoid freezing the camera feed.
    """
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def speak_async(engine, text):
    """
    Speaks the given text on a background thread so the camera feed
    and main loop remain fully responsive during speech.
    """
    print(f"Assistant: {text}")
    thread = threading.Thread(target=_speak_worker, args=(engine, text), daemon=True)
    thread.start()

def _speak_worker(engine, text):
    """Internal worker function run on a background thread."""
    engine.say(text)
    engine.runAndWait()
