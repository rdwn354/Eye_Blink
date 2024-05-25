import os
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

                self.total_elapsed_time = time.time() - self.last_time
                if self.total_elapsed_time >= self.minute_parameter:
                    self.rep += 1
                    self.last_time = time.time()

                class_label, output_data = self.classify_frame(frame)

                new_frame_time = time.time()
                self.fps = 1 / (new_frame_time - self.prev_frame_time)
                self.prev_frame_time = new_frame_time
                fps = str(self.fps)

                seconds = int(self.total_elapsed_time % 60)

                if class_label == 'close':
                    screenshot_name = f'screenshot_{self.frame_count}.png'
                    cv2.putText(frame, f"output data : {output_data}", (234, 88), self.font, self.size_font,
                                (255, 255, 255), 1)
                    cv2.putText(frame, f"status calculate : {class_label}", (234, 103), self.font, self.size_font,
                                (255, 255, 255), 1)
                    cv2.putText(frame, f"time : {self.rep}m {seconds}s", (234, 118), self.font, self.size_font,
                                (255, 255, 255), 1)
                    cv2.imwrite(screenshot_name, frame)
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
