# face_tracker_esp32.py — Jetson + RealSense face tracking
# Works WITH or WITHOUT the ESP32 connected.
#  - Shows the RealSense color display (if a display is available)
#  - Prints the error/angle command to the terminal every send
#  - If the ESP32 serial port is missing, it keeps running and just prints
#
# Quit: press 'q' in the window, or Ctrl+C in the terminal.

import sys
import glob
import time

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

try:
    import serial  # pyserial
except ImportError:
    serial = None

# ----------------------- Config -----------------------
W, H = 640, 480
BAUD = 115200
DEADBAND = 30            # px tolerance before sending a command
SEND_INTERVAL = 0.05     # seconds between sends (20 Hz max)
SHOW = True              # set False if running fully headless
MODEL_PATH = "yolov8n.pt"  # person detector (class 0). Swap for a face model if you have one.

# ----------------------- Serial (optional) -----------------------
def open_serial():
    """Try to auto-detect and open the ESP32. Returns a serial obj or None."""
    if serial is None:
        print("[serial] pyserial not installed — running in PRINT-ONLY mode.")
        return None

    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        print("[serial] No /dev/ttyUSB* or /dev/ttyACM* found — running in PRINT-ONLY mode.")
        return None

    port = ports[0]
    try:
        ser = serial.Serial(port, BAUD, timeout=0.05)
        time.sleep(2)  # ESP32 resets when port opens
        print(f"[serial] Connected to {port} @ {BAUD}")
        return ser
    except Exception as e:
        print(f"[serial] Failed to open {port}: {e} — running in PRINT-ONLY mode.")
        return None


def send_command(ser, error):
    """Send the pixel error to the ESP32 (if connected) and always print it."""
    msg = f"{error}\n"
    if ser is not None:
        try:
            ser.write(msg.encode())
        except Exception as e:
            print(f"[serial] write failed: {e}")
    # Always echo to terminal so you can verify behavior without hardware
    direction = "RIGHT" if error > 0 else "LEFT"
    print(f"[cmd] error={error:+4d} px  -> turn {direction}")


# ----------------------- Main -----------------------
def main():
    ser = open_serial()

    # RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print("[realsense] color stream started")

    model = YOLO(MODEL_PATH)
    print(f"[yolo] loaded {MODEL_PATH}")

    center_x = W // 2
    last_send = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            results = model(img, classes=[0], verbose=False)  # class 0 = person
            boxes = results[0].boxes

            error = None
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                x1, y1, x2, y2 = xyxy[areas.argmax()]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                error = cx - center_x

                if SHOW:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(img, f"err={error:+d}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                now = time.time()
                if abs(error) > DEADBAND and (now - last_send) > SEND_INTERVAL:
                    send_command(ser, error)
                    last_send = now

            if SHOW:
                cv2.line(img, (center_x, 0), (center_x, H), (255, 0, 0), 1)
                status = "SERIAL" if ser is not None else "PRINT-ONLY"
                cv2.putText(img, status, (W - 160, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                try:
                    cv2.imshow("Face Tracker", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    # No display available — fall back to headless automatically
                    print("[display] No display available, switching to headless.")
                    globals()["SHOW"] = False

    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C received.")
    finally:
        pipeline.stop()
        if ser is not None:
            ser.close()
        if SHOW:
            cv2.destroyAllWindows()
        print("[exit] cleaned up.")


if __name__ == "__main__":
    main()
