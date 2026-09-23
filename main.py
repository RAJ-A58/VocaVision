import cv2
from audio_output import initialize_tts, speak, speak_async
from camera_handler import show_camera_feed
from vision_engine import load_models, analyze_image

def main():
    try:
        # Initialize TTS engine and load both local neural network models
        tts_engine = initialize_tts()
        food_model, clothing_model, using_pytorch = load_models()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("VocaVision is starting...")
    # Synchronous speak before the camera loop — fine to block here
    speak(tts_engine, "VocaVision started. Press Space to analyze what is in front of the camera, or Q to quit.")

    try:
        for frame in show_camera_feed():          # generator yields only the frame
            cv2.imshow("VocaVision Feed", frame)

            # Poll for key press every 1 ms — keeps window responsive
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Non-blocking speech — camera feed stays live during inference
                speak_async(tts_engine, "Analyzing image...")
                print("Analyzing image...")

                try:
                    # analyze_image returns (description, annotated_frame, food_probs, clothing_probs)
                    description, *_ = analyze_image(food_model, clothing_model, frame,
                                                    using_pytorch=using_pytorch)
                    speak_async(tts_engine, description)
                except Exception as e:
                    print(f"Analysis Error: {e}")
                    speak_async(tts_engine, "Sorry, I encountered an error while analyzing the image.")

            elif key == ord('q'):
                print("Quitting VocaVision.")
                break

    except RuntimeError as e:
        print(f"Camera Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
