import os
import csv
import cv2
import time
import shutil
import numpy as np
import tensorflow as tf

class BlinkDetector:
    def __init__(self, folder_path, threshold, fps=15, minute_parameter=60, size_font=0.35):
        self.folder_path = folder_path
        self.threshold = threshold
        self.fps = fps
        self.size_font = size_font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.minute_parameter = minute_parameter
        self.status = False
        self.start_blink = None
        self.blink = []
        self.blink1D = []
        self.blink2D = [[0] * 50 for _ in range(240)]
        self.cek_blink = 0
        self.blink_row = 0
        self.frame_count = 0
        self.prev_frame_time = time.time()
        self.total_elapsed_time = 0
        self.close_folder_count = 0

        self.interpreter = tf.lite.Interpreter(model_path="Model/Model7_opt.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.path_save = f'Record/calculate'
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)

        self.last_time = time.time()
        self.output_folder = f"Record/images/{name}_folder_{self.close_folder_count}"
        os.makedirs(self.output_folder, exist_ok=True)

    def update_2d_array(self, csv_path):
        for i in range(min(len(self.blink1D), len(self.blink2D[0]))):
            self.blink2D[self.blink_row][i] = self.blink1D[i]
        self.blink1D = []
        self.blink_row = (self.blink_row + 1) % 240

        with open(csv_path, 'w', newline='') as csvfile:
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

    def process_video(self, video_path, csv_path, output_file):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_file, fourcc, self.fps, (448, 224))  # Adjusted output size
        
        self.close_folder_count += 1
        self.output_folder = f"Record/images/{name}_time_{self.close_folder_count}"
        os.makedirs(self.output_folder, exist_ok=True)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                class_label, output_data = self.classify_frame(frame)
                
                self.total_elapsed_time = time.time() - self.last_time
                if self.total_elapsed_time >= self.minute_parameter:
                    self.cek_blink = len(self.blink1D)
                    self.status = self.cek_blink <= 10
                    self.last_time = time.time()

                if class_label == "close":
                    self.start_blink = time.time()
                    cv2.rectangle(frame, (30, 30), (194, 194), (0, 255, 0), 1)
                    screenshot_name = f"{self.output_folder}/screenshot_{self.frame_count}.png"
                    cv2.imwrite(screenshot_name, frame)         

                elif class_label == "open":
                    self.frame_count += 1
                    if self.start_blink is not None:
                        blinkDur = time.time() - self.start_blink
                        self.blink1D.append(blinkDur)
                        self.start_blink = None

                new_frame_time = time.time()
                fps = 1 / (new_frame_time - self.prev_frame_time)
                self.prev_frame_time = new_frame_time

                cv2.rectangle(frame, (224, 60), (447, 223), (255, 255, 255), 1)
                cv2.putText(frame, f"status CVS : {'CVS' if self.status else 'Normal'}", (234, 73), self.font, self.size_font, (255, 255, 255), 1)
                cv2.putText(frame, f"output data : {output_data}", (234, 88), self.font, self.size_font, (255, 255, 255), 1)
                cv2.putText(frame, f"status calculate : {class_label}", (234, 103), self.font, self.size_font, (255, 255, 255), 1)
                cv2.putText(frame, f"fps : {int(fps)}", (234, 118), self.font, self.size_font, (255, 255, 255), 1)

                output.write(frame)
                cv2.imshow('Frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            output.release()
            cv2.destroyAllWindows()

            self.update_2d_array(csv_path)
            print("2D Array after storing:", self.blink2D)

        except Exception as e:
            print(f"An error occurred: {e}")
            cap.release()
            output.release()
            cv2.destroyAllWindows()

    def run_detection_on_folder(self):
        video_files = [f for f in os.listdir(self.folder_path) if f.endswith('.mp4')]
        video_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        for video_file in video_files:
            video_path = os.path.join(self.folder_path, video_file)
            csv_path = os.path.join('Record/csv', f"{os.path.basename(video_file).split('.')[0]}.csv")
            output_file = os.path.join(self.path_save, f"{os.path.basename(video_file).split('.')[0]}_calculate.mp4")

            self.process_video(video_path, csv_path, output_file)

    def backup_files(self):
        path_backup = f"Record/backup/{self.folder_path.split('/')[-1]}"
        if not os.path.exists(path_backup):
            os.mkdir(path_backup)

        for item in os.listdir(self.path_save):
            shutil.move(os.path.join(self.path_save, item), path_backup)
        for item in os.listdir('Record/csv'):
            shutil.move(os.path.join('Record/csv', item), path_backup)
        for item in os.listdir('Record/images'):
            shutil.move(os.path.join('Record/images', item), path_backup)


print("E-Blink".center(50, "-"))
name = input("input folder path : ")
folder_path = f"Record/capture/{name}"
threshold = float(input("input threshold : "))
blink_detector = BlinkDetector(folder_path, threshold)
blink_detector.run_detection_on_folder()
blink_detector.backup_files()
