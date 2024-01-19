import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.signal import fft2d, ifft2d, fftshift
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

base_path = r'C:\Workspace\CV_project\dataa'  
train_dir = os.path.join(base_path, 'train')
print('train data dir', train_dir)

img_width, img_height = 128, 128
batch_size = 256


 
if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:
    print("Please install GPU version of TF")
    

gpu_index = 0  # Specify the index of the GPU device you want to use
gpu_device = tf.config.list_physical_devices('GPU')[gpu_index]
tf.config.set_visible_devices(gpu_device, 'GPU')


def preprocess_image(image):
    # Split the image into RGB channels
    r, g, b = cv2.split(image)

    # Apply Fast Fourier Transform to each channel
    r_fft = fftshift(np.fft.fft2(r))
    g_fft = fftshift(np.fft.fft2(g))
    b_fft = fftshift(np.fft.fft2(b))

    # Normalize the transformed data
    r_normalized = np.abs(r_fft) / np.max(np.abs(r_fft))
    g_normalized = np.abs(g_fft) / np.max(np.abs(g_fft))
    b_normalized = np.abs(b_fft) / np.max(np.abs(b_fft))

    # Replace original channels with transformed and normalized channels
    image_transformed = cv2.merge((r_normalized, g_normalized, b_normalized))
    
    return image_transformed


image_generator = ImageDataGenerator(validation_split=0.25, preprocessing_function=preprocess_image) 

   

train_flow = image_generator.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    subset = "training",
    shuffle= True,
    batch_size=batch_size,
    class_mode='binary', # changed this from categorical
   
)

valid_flow = image_generator.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    subset = "validation",
    shuffle= True,
    batch_size=batch_size,
    class_mode='binary', # changed this from categorical
   
)


def build_model():
    initializer = tf.keras.initializers.GlorotNormal()
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_initializer=initializer))
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu', kernel_initializer=initializer))
    model.add(layers.Dense(1, activation='sigmoid'))  # Replace with 2 classes: "fake" and "real" # binary not be 2
    
    # t allows the output of the classification layer (fully connected + softmax activation function) 
    # # to be close to a uniform distribution, which corresponds to a loss of -log(1/n_classes).

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001, epsilon=1e-07 ),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
    )    
    return model


model = build_model()

# Train the model
epochs = 20
steps_per_epoch = len(train_flow)



callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint( 'C:/Workspace/CV_project/', monitor='val_accuracy', save_best_only=True)
history = model.fit(
    train_flow, 
    validation_data=valid_flow,
    epochs=epochs,
    callbacks=[callback, checkpoint],
    verbose = True
    
)

plt.plot(range(1, epochs + 1), history.history['accuracy'])
plt.plot(range(1, epochs + 1), history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_accuracy.png')
# plt.show()

plt.plot(range(1, epochs + 1), history.history['loss'])
plt.plot(range(1, epochs + 1), history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_loss.png')
# plt.show()



save_model(model, "latest_fft.h5")