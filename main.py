import time
import cv2
import numpy as np
from ultralytics import YOLO
import winsound

# ---------- Load logo once ----------
LOGO_PATH = "drownalertlogo.png"
logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
if logo is None:
    print(f"Warning: Could not load logo file: {LOGO_PATH}")


def overlay_logo(frame, logo_img, position="bottom-right", scale=0.18, opacity=1.0):
    """
    Overlay an RGBA (preferred) or RGB logo onto a BGR frame.
    Supported positions:
      - "top-left", "top-right", "bottom-left", "bottom-right"
      - "top-center", "center"
    scale = logo width as fraction of frame width
    opacity = 0..1 overall logo opacity
    Returns: (x, y, w, h) of where logo was drawn, or None if not drawn.
    """
    if logo_img is None or opacity <= 0:
        return None

    fh, fw = frame.shape[:2]

    target_w = max(1, int(fw * scale))
    target_h = max(1, int(logo_img.shape[0] * (target_w / logo_img.shape[1])))
    logo_rs = cv2.resize(logo_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if logo_rs.shape[2] == 4:
        logo_rgb = logo_rs[:, :, :3].astype(np.float32)
        alpha = (logo_rs[:, :, 3].astype(np.float32) / 255.0) * opacity
    else:
        logo_rgb = logo_rs.astype(np.float32)
        alpha = np.full((target_h, target_w), opacity, dtype=np.float32)

    pad = 12

    if position == "top-left":
        x, y = pad, pad
    elif position == "top-right":
        x, y = fw - target_w - pad, pad
    elif position == "bottom-left":
        x, y = pad, fh - target_h - pad
    elif position == "bottom-right":
        x, y = fw - target_w - pad, fh - target_h - pad
    elif position == "top-center":
        x, y = (fw - target_w) // 2, pad
    elif position == "center":
        x, y = (fw - target_w) // 2, (fh - target_h) // 2
    else:
        x, y = fw - target_w - pad, fh - target_h - pad

    if x < 0 or y < 0 or x + target_w > fw or y + target_h > fh:
        return None

    roi = frame[y:y + target_h, x:x + target_w].astype(np.float32)

    alpha_3 = np.dstack([alpha, alpha, alpha])
    blended = roi * (1 - alpha_3) + logo_rgb * alpha_3

    frame[y:y + target_h, x:x + target_w] = blended.astype(np.uint8)
    return (x, y, target_w, target_h)


# ---------- Splash screen ----------
_splash_clicked = False
_splash_logo_rect = None  # (x, y, w, h)

def _splash_mouse(event, x, y, flags, param):
    global _splash_clicked, _splash_logo_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        if _splash_logo_rect is None:
            # If we couldn't compute the rect, treat any click as continue
            _splash_clicked = True
            return
        lx, ly, lw, lh = _splash_logo_rect
        if lx <= x <= lx + lw and ly <= y <= ly + lh:
            _splash_clicked = True


def show_splash_screen(window_name="DrownAlert"):
    """
    White screen with centered logo. Continue on Enter/Space or clicking the logo.
    """
    global _splash_clicked, _splash_logo_rect
    _splash_clicked = False
    _splash_logo_rect = None

    W, H = 960, 540  # splash size (looks nice on Windows)
    splash = np.full((H, W, 3), 255, dtype=np.uint8)

    # Draw logo centered on white background
    _splash_logo_rect = overlay_logo(splash, logo, position="center", scale=0.42, opacity=1.0)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _splash_mouse)

    while True:
        cv2.imshow(window_name, splash)
        key = cv2.waitKey(30) & 0xFF

        # Enter or Space
        if key in (13, 32):
            break

        # ESC to exit program quickly (optional)
        if key == 27:
            cv2.destroyWindow(window_name)
            return False

        if _splash_clicked:
            break

    cv2.destroyWindow(window_name)
    return True


# -------------------- Polygon drawing state --------------------
poly_points = []

def polygon_mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        poly_points.append((x, y))

def draw_polygon_overlay(img, points, closed=False):
    if len(points) == 0:
        return
    for p in points:
        cv2.circle(img, p, 4, (0, 255, 0), -1)
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(img, [pts], closed, (0, 255, 0), 2)

def select_polygon_live(cap, window_name="Set Pool Boundary (Polygon)"):
    """
    Live-feed polygon selection:
      - Left click: add point
      - z: undo
      - c: clear
      - Enter/Space: confirm (>=3 points)
      - Esc: cancel
    """
    global poly_points
    poly_points = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, polygon_mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.destroyWindow(window_name)
            return None

        display = frame.copy()
        draw_polygon_overlay(display, poly_points, closed=False)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None
        if key == ord('z') and poly_points:
            poly_points.pop()
        if key == ord('c'):
            poly_points = []
        if key in (13, 32) and len(poly_points) >= 3:
            poly = np.array(poly_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.destroyWindow(window_name)
            return poly

def polygon_bbox(poly):
    xs = poly[:, 0, 0]
    ys = poly[:, 0, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def clamp_bbox(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def main():
    # -------------------- Settings --------------------
    CONF_THRESH = 0.6
    ALARM_AFTER_SECONDS = 5.0
    CLEAR_AFTER_SECONDS = 0.7

    BEEP_FREQ_HZ = 1200
    BEEP_DURATION_MS = 250
    BEEP_INTERVAL_SECONDS = 0.8

    # Flash settings for logo (bigger + prominent)
    FLASH_PERIOD = 0.35
    FLASH_LOGO_SCALE = 0.28          # bigger than normal
    FLASH_LOGO_POSITION = "top-center"

    # -------------------- Setup --------------------
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 1) Splash screen
    ok = show_splash_screen(window_name="DrownAlert")
    if not ok:
        cap.release()
        cv2.destroyAllWindows()
        return

    # 2) Live polygon selection
    poly = select_polygon_live(cap)
    if poly is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    system_active = False
    person_present_start = None
    alarm_active = False
    last_beep_time = 0.0
    last_seen_in_zone_time = 0.0

    cv2.namedWindow("YOLOv8 Live Person Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        display = frame.copy()

        # Boundary changes color based on system state
        boundary_color = (0, 0, 255) if system_active else (255, 255, 255)
        cv2.polylines(display, [poly], True, boundary_color, 2)

        # Crop around polygon bbox for speed
        bx1, by1, bx2, by2 = polygon_bbox(poly)
        bx1, by1, bx2, by2 = clamp_bbox(bx1, by1, bx2, by2, w, h)

        roi_frame = frame[by1:by2, bx1:bx2]
        offset_x, offset_y = bx1, by1

        results = model(roi_frame, verbose=False)
        person_in_zone_this_frame = False

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label != "person" or conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += offset_x
                x2 += offset_x
                y1 += offset_y
                y2 += offset_y

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) < 0:
                    continue

                person_in_zone_this_frame = True
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.time()

        # Update last seen time
        if person_in_zone_this_frame:
            last_seen_in_zone_time = now

        # Alarm logic (auto-off when person leaves)
        if system_active:
            if not alarm_active:
                if person_in_zone_this_frame:
                    if person_present_start is None:
                        person_present_start = now
                    if now - person_present_start >= ALARM_AFTER_SECONDS:
                        alarm_active = True
                        last_beep_time = 0.0
                else:
                    person_present_start = None
            else:
                # alarm is active; turn off after CLEAR_AFTER_SECONDS without seeing a person
                if now - last_seen_in_zone_time >= CLEAR_AFTER_SECONDS:
                    alarm_active = False
                    person_present_start = None
        else:
            alarm_active = False
            person_present_start = None

        # Beep while alarm active
        if alarm_active and now - last_beep_time >= BEEP_INTERVAL_SECONDS:
            winsound.Beep(BEEP_FREQ_HZ, BEEP_DURATION_MS)
            last_beep_time = now

        # Flash logo ONLY when alarm is active (bigger + prominent)
        if alarm_active:
            show_logo = (now % (2 * FLASH_PERIOD)) < FLASH_PERIOD
            if show_logo:
                overlay_logo(display, logo, position=FLASH_LOGO_POSITION,
                             scale=FLASH_LOGO_SCALE, opacity=1.0)

        cv2.imshow("YOLOv8 Live Person Detection", display)

        # Keybinds
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            system_active = not system_active
        elif key == ord('r'):
            new_poly = select_polygon_live(cap)
            if new_poly is not None:
                poly = new_poly
                alarm_active = False
                person_present_start = None
                last_seen_in_zone_time = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
