import cv2
import time

# Path to the video file
video_path = '../Record/capture/tes_time4.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the interval in seconds for taking screenshots
interval = 10

# Initialize variables
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Check if it's time to take a screenshot
    if elapsed_time >= interval:
        # Save the screenshot
        cv2.putText(frame, str(elapsed_time), (230, 80), cv2.FONT_HERSHEY_SIMPLEX , 0.35, (0, 255, 0), 1)
        screenshot_name = f'screenshot_{frame_count}.png'
        cv2.imwrite(screenshot_name, frame)

        # Reset the start time for the next interval
        start_time = time.time()
        frame_count += 1

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
