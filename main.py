import cv2

from ultralytics import YOLO



def main():
    #loads YOLOv8 model
    model = YOLO("yolov8n.pt")

    #starts webcam stream
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        # Load YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Start webcam stream
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Set resolution (optional, for smoother video)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create a resizable window
        cv2.namedWindow("YOLOv8 Live Person Detection", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run YOLOv8 inference
            results = model(frame, verbose=False)

            # Process detections
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    if label == "person":
                        print(f"Person detected with confidence {conf:.2f}")

                    # Draw bounding box and label
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the live video feed
            cv2.imshow("YOLOv8 Live Person Detection", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()





main()












