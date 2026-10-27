"""
VocaVision — Terminal-Based Vision & Speech Assistive System
============================================================
A 100% terminal-based application for assistive vision:
identifies food and clothing items, determines colors, and describes
them via audio speech (pyttsx3) and formatted console output.

No GUI or display windows required.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from audio_output import initialize_tts, speak, speak_async
from camera_handler import capture_frame
from vision_engine import (
    load_models, analyze_image,
    FOOD_CLASSES, CLOTHING_CLASSES,
)

BANNER = r"""
 __      __                 ____   ____.__       .__               
 \ \    / /___   ____ _____ \   \ /   /|__| _____|__| ____   ____  
  \ \  / //  _ \_/ ___\\__  \ \   Y   / |  |/  ___/  |/  _ \ /    \ 
   \ \/ /(  <_> )  \___ / __ \_\     /  |  |\___ \|  (  <_> )   |  \
    \__/  \____/ \___  >____  / \___/   |__|____  >__|\____/|___|  /
                     \/     \/                  \/               \/ 
             AI-Powered Assistive Vision [Terminal Edition]
"""

class VocaVisionTerminal:
    def __init__(self, enable_tts=True):
        self.tts_enabled = enable_tts
        self.tts_engine = None

        if self.tts_enabled:
            try:
                self.tts_engine = initialize_tts()
            except Exception as e:
                print(f"[!] Warning: TTS initialization failed ({e}). Proceeding in muted mode.")
                self.tts_enabled = False

        print("[*] Loading neural network models...")
        (
            self.food_model,
            self.clothing_model,
            self.using_pytorch,
            self.yolo_model,
            self.using_yolo,
        ) = load_models()

        if self.using_yolo:
            self.engine_name = "YOLO v3 (Custom Trained 20-Class Detector)"
        elif self.using_pytorch:
            self.engine_name = "Hybrid v2 (PyTorch MobileNetV2 + TF Food)"
        else:
            self.engine_name = "Legacy v1 (TensorFlow CNN)"

    def speak_text(self, text: str, async_mode: bool = True):
        """Speaks text if TTS is enabled, and prints message."""
        if not self.tts_enabled or self.tts_engine is None:
            return
        try:
            if async_mode:
                speak_async(self.tts_engine, text)
            else:
                speak(self.tts_engine, text)
        except Exception as e:
            print(f"[!] Speech error: {e}")

    def format_bar(self, prob: float, width: int = 20) -> str:
        """Renders an ASCII bar for confidence display."""
        filled = int(round(prob * width))
        return "█" * filled + "░" * (width - filled)

    def analyze_frame_data(self, frame: np.ndarray, save_path: str = None) -> str:
        """Runs model inference on an OpenCV BGR frame and displays terminal results."""
        h, w = frame.shape[:2]
        print(f"[*] Processing image ({w}x{h})...")

        start_t = time.perf_counter()
        description, annotated, food_probs, clothing_probs = analyze_image(
            self.food_model,
            self.clothing_model,
            frame,
            using_pytorch=self.using_pytorch,
            yolo_model=self.yolo_model,
            using_yolo=self.using_yolo,
        )
        latency = (time.perf_counter() - start_t) * 1000

        print("\n" + "═" * 68)
        print("  ANALYSIS RESULT")
        print("═" * 68)
        print(f"  Description : {description}")
        print(f"  Latency     : {latency:.1f} ms  | Engine: {self.engine_name}")
        print("═" * 68)

        if not self.using_yolo:
            print("\n  [Top Food Predictions]")
            for idx in np.argsort(food_probs)[::-1][:3]:
                bar = self.format_bar(food_probs[idx], width=18)
                print(f"    {FOOD_CLASSES[idx]:<16} {food_probs[idx]:>6.1%}  {bar}")

            print("\n  [Top Clothing Predictions]")
            for idx in np.argsort(clothing_probs)[::-1][:3]:
                bar = self.format_bar(clothing_probs[idx], width=18)
                print(f"    {CLOTHING_CLASSES[idx]:<16} {clothing_probs[idx]:>6.1%}  {bar}")
            print("-" * 68)

        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"[+] Saved annotated frame to: {save_path}")

        # Speak result
        self.speak_text(description, async_mode=True)
        return description

    def analyze_image_file(self, file_path: str, save_path: str = None):
        """Analyzes an image from file path."""
        if not os.path.isfile(file_path):
            print(f"[-] Error: File not found: '{file_path}'")
            return

        frame = cv2.imread(file_path)
        if frame is None:
            print(f"[-] Error: Unable to decode image '{file_path}'.")
            return

        out_path = save_path or (os.path.splitext(file_path)[0] + "_result.jpg")
        self.analyze_frame_data(frame, save_path=out_path)

    def analyze_webcam_snapshot(self, camera_index: int = 0, save_path: str = None):
        """Snaps a single frame from webcam and analyzes it."""
        print(f"[*] Accessing camera at index {camera_index}...")
        try:
            frame = capture_frame(camera_index)
        except Exception as e:
            print(f"[-] Camera Error: {e}")
            self.speak_text("Could not open camera.")
            return

        out_path = save_path or "webcam_latest_result.jpg"
        self.analyze_frame_data(frame, save_path=out_path)

    def run_continuous_monitor(self, camera_index: int = 0, interval: float = 2.0):
        """
        Periodically captures a frame from the webcam and speaks descriptions
        only when scene content changes. Fully headless terminal loop.
        """
        print(f"\n[*] Starting Continuous Camera Monitor (Index: {camera_index}, Interval: {interval}s)")
        print("[*] Press Ctrl+C at any time in the terminal to stop monitoring.\n")
        self.speak_text("Continuous camera monitoring started.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[-] Failed to open webcam at index {camera_index}.")
            return

        last_description = ""
        scan_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[-] Failed to read frame from camera.")
                    time.sleep(interval)
                    continue

                scan_count += 1
                description, annotated, _, _ = analyze_image(
                    self.food_model,
                    self.clothing_model,
                    frame,
                    using_pytorch=self.using_pytorch,
                    yolo_model=self.yolo_model,
                    using_yolo=self.using_yolo,
                )

                timestamp = time.strftime("%H:%M:%S")
                if description != last_description:
                    print(f"[{timestamp}] [Scan #{scan_count}] -> {description}")
                    self.speak_text(description, async_mode=True)
                    last_description = description
                    # Save snapshot of new detection
                    cv2.imwrite("monitor_latest.jpg", annotated)
                else:
                    # Heartbeat log on same scene
                    print(f"[{timestamp}] [Scan #{scan_count}] (Unchanged) {description}", end="\r")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[*] Monitoring stopped by user.")
            self.speak_text("Camera monitoring stopped.")
        finally:
            cap.release()

    def interactive_menu(self):
        """Interactive terminal menu when run without CLI flags."""
        print(BANNER)
        tts_status = "ENABLED" if self.tts_enabled else "MUTED"
        print(f"  Active Engine : {self.engine_name}")
        print(f"  Voice Output  : {tts_status}")
        print("═" * 68)

        self.speak_text("VocaVision is ready.")

        while True:
            print("\nSelect an action:")
            print("  [1] Snap & Analyze Webcam (Single Shot)")
            print("  [2] Continuous Camera Monitor (Background scan & voice alerts)")
            print("  [3] Analyze Image File")
            print(f"  [4] Toggle Voice Output (Currently {tts_status})")
            print("  [5] Exit")

            try:
                choice = input("\nEnter choice [1-5]: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting VocaVision.")
                break

            if choice == "1":
                self.analyze_webcam_snapshot()
            elif choice == "2":
                try:
                    int_str = input("Scan interval in seconds [default: 2.0]: ").strip()
                    interval = float(int_str) if int_str else 2.0
                except ValueError:
                    interval = 2.0
                self.run_continuous_monitor(interval=interval)
            elif choice == "3":
                path = input("Enter path to image file: ").strip().strip('"').strip("'")
                if path:
                    self.analyze_image_file(path)
            elif choice == "4":
                self.tts_enabled = not self.tts_enabled
                tts_status = "ENABLED" if self.tts_enabled else "MUTED"
                if self.tts_enabled and self.tts_engine is None:
                    try:
                        self.tts_engine = initialize_tts()
                    except Exception as e:
                        print(f"[-] TTS error: {e}")
                        self.tts_enabled = False
                        tts_status = "MUTED"
                print(f"[+] Voice output is now {tts_status}.")
            elif choice == "5":
                print("\nThank you for using VocaVision. Goodbye!")
                self.speak_text("Goodbye.")
                break
            else:
                print("[-] Invalid choice. Please enter a number from 1 to 5.")


def main():
    parser = argparse.ArgumentParser(
        description="VocaVision: Terminal-Based AI Vision & Speech Assistive System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, help="Path to an image file to analyze")
    parser.add_argument("--webcam", action="store_true", help="Snap a single photo from webcam and analyze")
    parser.add_argument("--monitor", action="store_true", help="Run continuous headless webcam monitoring loop")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval (seconds) between scans in monitor mode")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--save", type=str, default=None, help="File path to save the annotated detection image")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech audio announcements")

    args = parser.parse_args()

    app = VocaVisionTerminal(enable_tts=not args.no_tts)

    if args.image:
        app.analyze_image_file(args.image, save_path=args.save)
    elif args.webcam:
        app.analyze_webcam_snapshot(camera_index=args.camera, save_path=args.save)
    elif args.monitor:
        app.run_continuous_monitor(camera_index=args.camera, interval=args.interval)
    else:
        # No CLI flags -> launch interactive menu
        app.interactive_menu()


if __name__ == "__main__":
    main()
