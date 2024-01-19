#libraries
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
import os
import imageio
from sklearn.model_selection import train_test_split
from keras import layers, models


# Intial Variables
dataset_path = "dataset"
train_data = []
train_labels = []
test_data = []
test_labels = []
image_size = (256, 256)

# Train Test Split
for filename in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, filename)
    image = imageio.imread(image_path, as_gray=True)
    image_dct = cv2.dct(image)
    image_array = np.array(image_dct)
    resized_image = cv2.resize(image_array, image_size)
    train_data.append(resized_image)
    if filename.endswith('.jpg'):
        train_labels.append(0)  
    elif filename.endswith('.png'):
        train_labels.append(1)

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)



#Scaling
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)
train_labels = train_labels.reshape(-1,)
test_labels = test_labels.reshape(-1,)
train_data = train_data / 255.0
test_data = test_data / 255.0


# Initialize CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=20, callbacks=[early_stopping])
score = model.evaluate(test_data, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save('freq_analysis_model.h5')

# Testing
image = imageio.imread('1.jpg', as_gray=True)
image_dct = cv2.dct(image)
image_dct = cv2.resize(image_dct, image_size)
image_dct = np.array([image_dct]) / 255.0
prediction = model.predict(image_dct)
if prediction > 0.5:
    print('GAN')
else:
    print('Real')





