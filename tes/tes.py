import os
import shutil
import cv2
import time
import numpy as np
import tensorflow as tf


class BlinkDetector:
    def __init__(self, name, threshold, fps=15, minute_parameter=60, size_font=0.35):
        self.name = name
        self.threshold = threshold
        self.fps = fps
        self.minute_parameter = minute_parameter
        self.size_font = size_font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.status = False
        self.total_elapsed_time = 0
        self.rep = 0
        self.frame_count = 0
        self.prev_frame_time = 0
        self.last_close_time = 0
        self.close_folder_count = 1
        self.prev_time_write = 0

        self.interpreter = tf.lite.Interpreter(model_path="../Model/Model7_opt.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.path_video = f"../Record/capture/{name}.mp4"

        if not os.path.exists(self.path_video):
            raise FileNotFoundError(f"Video file {self.path_video} not found")

        self.cap = cv2.VideoCapture(self.path_video)
        self.fps_cap = self.cap.get(cv2.CAP_PROP_FPS)
        self.last_time = time.time()

        # Create initial folder to save images_tes_time
        self.output_folder = f"images/{name}_folder_{self.close_folder_count}"
        os.makedirs(self.output_folder, exist_ok=True)

    def preprocess_frame(self, frame):
        normalized_frame = frame / 255.0
        preprocessed_frame = np.expand_dims(normalized_frame.astype(np.float32), axis=0)
        return preprocessed_frame

    def classify_frame(self, frame):
        roi_frame = frame[:, :224, :]
        preprocessed_frame = self.preprocess_frame(roi_frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        class_label = "open" if output_data <= self.threshold else "close"
        return class_label, output_data

    def run_detection(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                class_label, output_data = self.classify_frame(frame)

                self.total_elapsed_time = time.time() - self.last_time
                if self.total_elapsed_time >= self.minute_parameter:
                    self.rep += 1
                    self.last_time = time.time()

                    # Create a new folder for saving images_tes_time every minute
                    self.close_folder_count += 1
                    self.output_folder = f"images/{self.name}_folder_{self.close_folder_count}"
                    os.makedirs(self.output_folder, exist_ok=True)

                new_frame_time = time.time()
                self.fps = 1 / (new_frame_time - self.prev_frame_time)
                self.prev_frame_time = new_frame_time
                fps = str(self.fps)

                seconds = int(self.total_elapsed_time % 60)

                time_write = time.time()
                if class_label == 'close':
                    screenshot_name = f'{self.output_folder}/screenshot_{self.frame_count}.png'
                    cv2.imwrite(screenshot_name, frame)

                elif class_label == 'open':
                    self.frame_count += 1

                cv2.putText(frame, f"output data : {output_data}", (234, 88), self.font, self.size_font,
                            (255, 255, 255), 1)
                cv2.putText(frame, f"status calculate : {class_label}", (234, 103), self.font, self.size_font,
                            (255, 255, 255), 1)
                cv2.putText(frame, f"time : {self.rep}m {seconds}s", (234, 118), self.font, self.size_font,
                            (255, 255, 255), 1)

                cv2.imshow('Calculate', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred : {e}")


print("E-Blink".center(50, "-"))
name = input("input name : ")
threshold = float(input("input threshold : "))
try:
    blink_detector = BlinkDetector(name, threshold)
    blink_detector.run_detection()
except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred : {e}")
