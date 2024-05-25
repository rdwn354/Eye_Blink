import cv2
import time

cap = cv2.VideoCapture("Record/capture/Ridwan_ml.mp4")

start_time = time.time()  # Start time for the timer
rep = 0
slow_down_factor = 1.75

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)



    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:
        rep += 1



    # Add timer text to the frame
    cv2.putText(frame, f"time: {rep}m, {elapsed_time:.2f}s", (230, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imshow('frame', frame)

    cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS) / slow_down_factor))

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
