import cv2
from ultralytics import YOLO


def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # 🔹 Use the SAME index that worked in your test.
    # If your test script used index 1, change this to VideoCapture(1).
    cap = cv2.VideoCapture(0)  # <-- adjust index if needed

    # Optional: you can also try adding CAP_DSHOW on Windows:
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Grab one frame for ROI selection
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame from camera for ROI selection.")
        cap.release()
        return

    # 🔹 Create a window and let the user select the ROI on this still frame
    cv2.namedWindow("Select Pool Region", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select Pool Region", frame,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Pool Region")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No ROI selected. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Convert to (x1, y1, x2, y2)
    roi_x1, roi_y1 = x, y
    roi_x2, roi_y2 = x + w, y + h

    print(f"Selected ROI: ({roi_x1}, {roi_y1}) -> ({roi_x2}, {roi_y2})")

    cv2.namedWindow("YOLOv8 Live Person Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from camera during live feed.")
            break

        # Work on a copy so we keep the raw frame intact
        display = frame.copy()

        # Draw pool region (user-selected)
        cv2.rectangle(display, (roi_x1, roi_y1), (roi_x2, roi_y2),
                      (255, 255, 255), 2)
        cv2.putText(display, "Pool Region",
                    (roi_x1, max(roi_y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # YOLO inference
        # --- OPTION B: run YOLO only on the ROI crop ---

        # 1. Crop the region of interest
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # 2. Run YOLO on the cropped ROI
        results = model(roi_frame, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label != "person":
                    continue
                if conf < 0.6:
                    continue

                # YOLO gives coords relative to roi_frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 3. Shift them back to full-frame coordinates
                x1 += roi_x1
                x2 += roi_x1
                y1 += roi_y1
                y2 += roi_y1

                # Draw detection box (no need to check ROI – all detections are inside it)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"person {conf:.2f}",
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Live Person Detection", display)

        # Quit on 'q', reselect ROI on 'r'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reselect pool region using current frame as snapshot
            cv2.namedWindow("Select Pool Region", cv2.WINDOW_NORMAL)
            roi = cv2.selectROI("Select Pool Region", frame,
                                fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Pool Region")
            x, y, w, h = roi
            if w > 0 and h > 0:
                roi_x1, roi_y1 = x, y
                roi_x2, roi_y2 = x + w, y + h
                print(f"New ROI: ({roi_x1}, {roi_y1}) -> ({roi_x2}, {roi_y2})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
