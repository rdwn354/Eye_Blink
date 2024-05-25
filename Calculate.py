import os
import csv
import cv2
import time
import shutil
import numpy as np
import tensorflow as tf

class BlinkDetector:
    def __init__(self, name, threshold, fps=15, minute_parameter=60, size_font=0.35, slow_down = 20):
        self.name = name
        self.threshold = threshold
        self.fps = fps
        self.minute_parameter = minute_parameter
        self.size_font = size_font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.status = False
        self.start_blink = None
        self.blink = []
        self.blink1D = []
        self.blink2D = [[0] * 70 for _ in range(240)]
        self.blink_row = 0
        self.frame_count = 0
        self.total_elapsed_time = 0
        self.close_folder_count = 0
        self.prev_frame_time = 0
        self.rep = 0
        self.slow_down = slow_down
        self.cek_blink = 0

        self.interpreter = tf.lite.Interpreter(model_path="Model/Model7_opt.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.path_video = f'Record/capture/{self.name}.mp4'
        self.path_csv = f'Record/csv/{self.name}.csv'
        self.path_save = f'Record/calculate'

        if not os.path.exists(self.path_video):
            os.makedirs(self.path_video)

        self.cap = cv2.VideoCapture(self.path_video) # can change into real camera
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_file = os.path.join(self.path_save, f"{self.name}_calculate.mp4")
        self.output = cv2.VideoWriter(self.output_file, fourcc, self.fps, (448, 224))  # Adjusted output size
        self.last_time = time.time()

        self.output_folder = f"images/{name}_minute_{self.close_folder_count}"
        os.makedirs(self.output_folder, exist_ok=True)

    def update_2d_array(self):
        for i in range(min(len(self.blink1D), len(self.blink2D[0]))):
            self.blink2D[self.blink_row][i] = self.blink1D[i]
        self.blink1D = []
        self.blink_row = (self.blink_row + 1) % 240

        with open(self.path_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.blink2D)

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
            while (self.cap.isOpened()):
                ret, frame = self.cap.read()
                if not ret:
                    break

                class_label, output_data = self.classify_frame(frame)

                self.total_elapsed_time = time.time() - self.last_time
                if self.total_elapsed_time >= self.minute_parameter:
                    self.rep += 1

                    # Create a new folder for saving images_tes_time every minute
                    self.close_folder_count += 1
                    self.output_folder = f"images/{self.name}_minute_{self.close_folder_count}"
                    os.makedirs(self.output_folder, exist_ok=True)

                    self.cek_blink = len(self.blink1D)
                    if self.cek_blink <= 10:  # cek status cvs
                        self.status = True
                    else:
                        self.status = False

                    self.update_2d_array()
                    print("2D Array after storing:", self.blink2D)
                    self.last_time = time.time()

                if class_label == "close":
                    self.start_blink = time.time()
                    cv2.rectangle(frame, (30, 30), (194, 194), (255, 255, 255), 1)
                    screenshot_name = f'{self.output_folder}/screenshot_{self.frame_count}.png'
                    cv2.imwrite(screenshot_name, frame)

                elif class_label == "open":
                    self.frame_count += 1
                    if self.start_blink is not None:
                        blinkDur = time.time() - self.start_blink
                        self.blink1D.append(blinkDur)
                        self.start_blink = None

                new_frame_time = time.time()
                self.fps = 1 / (new_frame_time - self.prev_frame_time)
                self.prev_frame_time = new_frame_time
                fps = str(int(self.fps))

                seconds = int(self.total_elapsed_time % 60)

                cv2.rectangle(frame, (224, 60), (447, 223), (255, 255, 255), 1)
                cv2.putText(frame, f"status CVS : {'CVS' if self.status else 'Normal'}", (234, 73), self.font, self.size_font, (255,255,255), 1)
                cv2.putText(frame, f"output data : {output_data}", (234, 88), self.font, self.size_font, (255,255,255), 1)
                cv2.putText(frame, f"status calculate : {class_label}", (234, 103), self.font, self.size_font, (255,255,255), 1)
                cv2.putText(frame, f"time : {self.rep}m {seconds}s", (234, 118), self.font, self.size_font,(255, 255, 255), 1)
                cv2.putText(frame, f"blink detected per minutes : {self.cek_blink}", (234, 133), self.font, self.size_font, (255, 255, 255), 1)

                self.output.write(frame)
                cv2.imshow('Calculate', frame)
                cv2.waitKey(int(1000 / self.cap.get(cv2.CAP_PROP_FPS) / self.slow_down))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            self.output.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred : {e}")

    def backup_files(self):
        path_backup = f"Record/backup/{self.name}"
        if not os.path.exists(path_backup):
            os.mkdir(path_backup)

        dest_video = shutil.move(self.path_video, path_backup)
        dest_csv = shutil.move(self.path_csv, path_backup)
        dest_cal = shutil.move(self.path_save, path_backup)

        os.mkdir(self.path_save)


print("E-Blink".center(50, "-"))
name = input("input name : ")
threshold = float(input("input threshold : "))
blink_detector = BlinkDetector(name, threshold)
blink_detector.run_detection()
blink_detector.backup_files()
