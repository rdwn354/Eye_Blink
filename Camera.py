import os
import cv2
import time
import numpy as np
import tensorflow as tf

class EyeStatusDetector:
    def __init__(self, model_path="Model/Model7_opt.tflite", camera_index=1):
        self.size_font = 0.35
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.prev_frame_time = 0

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    def preprocess_frame(self, frame):
        normalized_frame = frame / 255.0
        preprocessed_frame = np.expand_dims(normalized_frame.astype(np.float32), axis=0)
        return preprocessed_frame

    def classify_frame(self, frame):
        preprocessed_frame = self.preprocess_frame(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        class_label = "open" if output_data <= 0.5 else "close"
        return class_label, output_data

    def run_detection(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
                class_label, output_data = self.classify_frame(frame)

                # Calculate frames per second
                current_time = time.time()
                fps = 1 / (current_time - self.prev_frame_time)
                self.prev_frame_time = current_time

                if class_label == "close":
                    cv2.rectangle(frame, (30, 30), (194, 194), (255, 255, 255), 1)

                # Display prediction result on the frame
                cv2.putText(frame, f"status: {class_label} ({output_data[0][0]:.2f})", (0, 20), self.font, self.size_font, (0, 255, 0), 1)
                cv2.putText(frame, f'fps: {fps:.2f}', (0, 40), self.font, self.size_font, (0, 255, 0), 1)

                cv2.imshow('Frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred: {e}")


detector = EyeStatusDetector()
detector.run_detection()
