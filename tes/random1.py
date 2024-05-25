import cv2
import time
import numpy as np
import tensorflow as tf
import os


class EyeStatusDetector:
    def __init__(self, model_path="../Model/Model7_opt.tflite", camera_index=1, yolo_weights="yolov3.weights",
                 yolo_cfg="yolov3.cfg", yolo_classes="yolov3.txt"):
        self.size_font = 0.35
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.prev_frame_time = 0

        # Ensure the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # Ensure YOLO files exist
        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(f"YOLO weights file not found: {yolo_weights}")
        if not os.path.exists(yolo_cfg):
            raise FileNotFoundError(f"YOLO config file not found: {yolo_cfg}")
        if not os.path.exists(yolo_classes):
            raise FileNotFoundError(f"YOLO classes file not found: {yolo_classes}")

        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        with open(yolo_classes, 'r') as f:
            self.classes = f.read().strip().split("\n")

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

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

    def detect_eyes(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == self.classes.index(
                        "person") and confidence > 0.5:  # Adjust as needed for "eye" detection
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        eyes = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                eyes.append((x, y, w, h))
        return eyes

    def run_detection(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                eyes = self.detect_eyes(frame)
                for (x, y, w, h) in eyes:
                    eye_region = frame[y:y + h, x:x + w]
                    eye_region = cv2.resize(eye_region,
                                            (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
                    class_label, output_data = self.classify_frame(eye_region)

                    color = (0, 255, 0) if class_label == "open" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"status: {class_label} ({output_data[0][0]:.2f})", (x, y - 10), self.font,
                                self.size_font, color, 1)

                current_time = time.time()
                fps = 1 / (current_time - self.prev_frame_time)
                self.prev_frame_time = current_time

                cv2.putText(frame, f'fps: {fps:.2f}', (10, 20), self.font, self.size_font, (0, 255, 0), 1)
                cv2.imshow('Frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred: {e}")


detector = EyeStatusDetector()
detector.run_detection()
