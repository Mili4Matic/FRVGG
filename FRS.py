# Author: Alejandro Milikowsky
# Email: alejandro.milikowsky@estudiante.uam.es
# Date: July 2024
#
# This program is developed for real-time face recognition using a webcam.
# It leverages the VGGFace model to extract facial features and compares them 
# with known individuals' features stored in a specified directory.
# The purpose is to identify known individuals and label them accordingly 
# while marking unknown faces. The program also updates the known features 
# with new captures if the recognition confidence is high.
#
# Directory Structure for Known Faces:
# The known directory should contain subdirectories named after known individuals.
# Each subdirectory should contain images of the person.
# Example:
# /home/mili/Desktop/FRTest/FRVGG/known/
# ├── person1/
# │   ├── image1.jpg
# │   ├── image2.jpg
# │   └── ...
# ├── person2/
# │   ├── image1.jpg
# │   ├── image2.png
# │   └── ...
# └── ...

from keras.engine import Model
from keras.layers import Flatten
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.preprocessing import image
from keras_vggface import utils
import cv2
import os
from scipy.spatial.distance import cosine

# loading only feature extractor layers (check documentation: https://github.com/rcmalli/keras-vggface)
vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: max, avg, min

# cv Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocessing model function with our faces on our directories
def preprocess_image(img):
    img = cv2.resize(img, (224, 224)).astype('float32')
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=1)  # version=1 for VGGFace1, version=2 for VGGFace2, !!Version 1 for VGG16, Version 2 for RESNET AND SENET!!
    return img

# Feature extraction using vgg model from the image
def extract_features(model, img):
    features = model.predict(img)
    return features.flatten()  # 1D flattened

# Load person and known features
def load_known_features(known_images_dir):
    known_features = {}
    for person in os.listdir(known_images_dir):
        person_dir = os.path.join(known_images_dir, person)
        if os.path.isdir(person_dir):
            features_list = []
            for img_path in os.listdir(person_dir):
                img_full_path = os.path.join(person_dir, img_path)
                features = extract_features(vgg_model, preprocess_image(cv2.imread(img_full_path)))
                features_list.append(features)
            known_features[person] = features_list
    return known_features

# Check and label person
def classify_person(webcam_features, known_features, similarity_threshold=0.68):
    min_distance = float('inf')
    best_match = None
    for person, features_list in known_features.items():
        for features in features_list:
            distance = cosine(webcam_features, features)
            if distance < min_distance:
                min_distance = distance
                best_match = person
    known_person = min_distance < (1 - similarity_threshold)
    return known_person, best_match, min_distance

# Updating features
def update_known_features(known_features, person, new_features):
    known_features[person].append(new_features)

# Load known images features
known_images_dir = '/home/mili/Desktop/FRTest/FRVGG/known'
known_features = load_known_features(known_images_dir)

# Webcam function
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img_preprocessed = preprocess_image(face_img)
        face_features = extract_features(vgg_model, face_img_preprocessed)
        
        known, person, min_distance = classify_person(face_features, known_features)
        label = "Unknown"
        percentage = (1 - min_distance) * 100
        
        # Check if I know him, known if percentage is >65, update features with captured image if percentage is >80
        if known and percentage > 70:
            label = f'{person} ({percentage:.2f}%)'
            if percentage > 80:
                update_known_features(known_features, person, face_features)
        else:
            label = f'Unknown ({percentage:.2f}%)'

        # Rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if known else (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if known else (0, 0, 255), 2)

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
