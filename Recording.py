import os
import cv2
import time
import numpy as np
import tensorflow as tf


class VideoRecorder:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.size_font = 0.35
        self.interpreter = tf.lite.Interpreter(model_path="Model/Model7_opt.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_frame(self, frame):
        normalized_frame = frame / 255.0
        preprocessed_frame = np.expand_dims(normalized_frame.astype(np.float32), axis=0)
        return preprocessed_frame

    def classify_frame(self, frame):
        preprocessed_frame = self.preprocess_frame(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        class_label = "open" if output_data <= self.threshold else "close"
        return class_label, output_data

    def stackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver 

    def record_video(self, name, threshold, fps=15):
        print("E-Blink".center(50, "-"))
        self.threshold = threshold
        self.status = 0
        self.total_elapsed_time = 0
        self.prev_frame_time = 0
        self.rep = 0
        path_video = 'Record/capture'

        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not os.path.exists(path_video):
            os.makedirs(path_video)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = os.path.join(path_video, f"{name}.mp4")
        output = cv2.VideoWriter(output_file, fourcc, fps, (448, 224))  # Adjusted output size

        timer_now = time.time()
        last_time = timer_now

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.total_elapsed_time = time.time() - last_time
                if self.total_elapsed_time >= 60:
                    self.rep += 1
                    last_time = time.time()

                frame = cv2.resize(frame, (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
                class_label, output_data = self.classify_frame(frame)

                new_frame_time = time.time()
                fps = 1 / (new_frame_time - self.prev_frame_time)
                self.prev_frame_time = new_frame_time
                fps = str(int(fps))

                # Create a fresh blank image for each frame
                blank = np.zeros((224, 224, 3), dtype='uint8')

                seconds = int(self.total_elapsed_time % 60)

                # Display prediction result on the frame
                cv2.putText(blank, f'Name: {name}', (10, 10), self.font, self.size_font, (255, 255, 255), 1)
                cv2.putText(blank, f'fps: {fps}', (10, 25), self.font, self.size_font, (255, 255, 255), 1)
                cv2.putText(blank, f"status on recording : {class_label}", (10, 40), self.font, self.size_font, (255, 255, 255), 1)
                cv2.putText(blank, f"time: {self.rep}m {seconds}s", (10, 55), self.font, self.size_font, (255, 255, 255), 1)

                cv2.rectangle(blank, (0, 0), (223, 60), (255, 255, 255), 1)

                imStack = self.stackImages(1, [frame, blank])
                output.write(imStack)

                cv2.imshow('Frame', imStack)

                if self.rep >= 120:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            output.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred: {e}")


video_recorder = VideoRecorder()
name = input("input name : ")
threshold = float(input("input threshold : "))
video_recorder.record_video(name, threshold)
