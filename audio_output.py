import pyttsx3

def initialize_tts():
    """Initializes the pyttsx3 Text-to-Speech engine."""
    engine = pyttsx3.init()
    # Optional: adjust rate or volume here
    engine.setProperty('rate', 175)
    return engine

def speak(engine, text):
    """Speaks the given text out loud."""
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()
