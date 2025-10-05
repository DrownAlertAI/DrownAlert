import cv2

def main():
    # open webcam (0 = default camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    pic_count = 0
    max_pics = 10

    try:
        while True:
            ret, frame = cap.read()

            cv2.imshow('Webcam', frame)

            # press 's' to save a picture
            # press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and pic_count < max_pics:
                filename = f"snapshot_{pic_count+1}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                pic_count += 1
                if pic_count >= max_pics:
                    print("Reached max number of pictures (10).")
            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




