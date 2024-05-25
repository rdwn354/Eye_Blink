import os
import numpy as np
import tensorflow as tf
from time import time
from keras.preprocessing.image import load_img, img_to_array

class EyeClassifier:
    def __init__(self, model_path, threshold):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.threshold = threshold

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

def classify_eye_state(model_path, image_path, threshold):
    eye_classifier = EyeClassifier(model_path, threshold)
    image = load_img(image_path, target_size=(224, 224))  # Assuming images are resized to 224x224
    image_array = img_to_array(image)
    class_label, output_data = eye_classifier.classify_frame(image_array)
    return output_data, class_label


name_folder = "images"
model_quantization = "../Model/Model7_opt.tflite"

for folder_name in os.listdir(name_folder):
    folder_path = os.path.join(name_folder, folder_name)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Call your function to classify the eye state for each image
        output_data, class_label = classify_eye_state(model_quantization, image_path, 0.5)

        print("Image name:", image_name)
        print("Predicted class:", class_label)
        print("Output data:", output_data)
        print("")
