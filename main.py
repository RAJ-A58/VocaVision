import os
import cv2
from dotenv import load_dotenv
from audio_output import initialize_tts, speak, speak_async
from camera_handler import show_camera_feed
from vision_engine import get_genai_client, analyze_image

def main():
    # Load environment variables (e.g., GEMINI_API_KEY)
    load_dotenv()

    try:
        # Initialize components
        tts_engine = initialize_tts()
        genai_client = get_genai_client()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("VocaVision is starting...")
    # Use synchronous speak here (before camera loop starts) — fine to block
    speak(tts_engine, "VocaVision started. Press Space to analyze what is in front of the camera, or Q to quit.")

    # Start the camera feed
    camera_generator = show_camera_feed()

    try:
        for frame in camera_generator:  # generator now yields only frame
            # Display the camera feed
            cv2.imshow("VocaVision Feed", frame)

            # Wait for key press (1ms — keeps the window responsive)
            key = cv2.waitKey(1) & 0xFF

            # If 'Space' is pressed, capture and analyze
            if key == ord(' '):
                # Non-blocking — camera feed stays live during speech & API call
                speak_async(tts_engine, "Analyzing image...")
                print("Analyzing image...")

                try:
                    # Analyze the current frame
                    description = analyze_image(genai_client, frame)

                    # Speak result on background thread — camera feed stays responsive
                    speak_async(tts_engine, description)
                except Exception as e:
                    error_msg = f"Failed to analyze image: {e}"
                    print(error_msg)
                    speak_async(tts_engine, "Sorry, I encountered an error while analyzing the image.")

            # If 'Q' is pressed, quit the application
            elif key == ord('q'):
                print("Quitting VocaVision.")
                break

    except RuntimeError as e:
        print(f"Camera Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
