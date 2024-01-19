import dlib
import cv2
import numpy as np
import os

# Load the face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/Rakesh/cvproj/shape_predictor_68_face_landmarks.dat")

# Define the folder containing the images
folder_path = "D:/Rakesh/cvproj/test_real"  # Replace with your own folder path

# Initialize an empty list to store the feature vectors
features = []

# Iterate over each image in the folder
for filename in os.listdir(folder_path):
    # Load the input image

    image = cv2.imread(os.path.join(folder_path, filename))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the face in the grayscale image
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        # Predict the landmarks for the detected face
        landmarks = predictor(gray, face)
        pts = np.zeros((51, 2))
        for i in range(51):
            pts[i] = (landmarks.part(i+17).x, landmarks.part(i+17).y)
        warped_pts = pts
        # Normalize the warped landmarks to the [0, 1] x [0, 1] region
        warped_pts[:, 0] /= image.shape[1]
        warped_pts[:, 1] /= image.shape[0]
        # Concatenate the x-coordinates and y-coordinates into a single feature vector
        feature_vector = np.concatenate([warped_pts[:, 0], warped_pts[:, 1]])

        # Append the feature vector to the list of features
        features.append(feature_vector)

# Convert the list of features to a numpy array
features = np.array(features)
np.save("features_test.npy", features)
