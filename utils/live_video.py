import cv2
global frame

def get_live_feed():
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Live Video Feed', frame)

        # Press 'q' on keyboard to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_live_feed()
