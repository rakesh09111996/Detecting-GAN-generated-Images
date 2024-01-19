import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from tensorflow.keras.models import save_model

train_dir = r'C:\Users\harin\OneDrive\Documents\Archana\Indiana\Academics\2_Semester\CV\Project\Code\data'  
   
def compute_spatial_cooccurrence(channel, delta):
    # Compute the spatial co-occurrence matrix for a single channel
    num_bins = 256
    N, M = channel.shape[:2]
    cooccurrence_matrix = np.zeros((num_bins, num_bins), dtype=np.int32)
    for x in range(N):
        for y in range(M):
            i = channel[x, y]
            if x + delta[0] < N and y + delta[1] < M:
                j = channel[x + delta[0], y + delta[1]]
                cooccurrence_matrix[int(i), int(j)] += 1
    return cooccurrence_matrix

def compute_crossband_cooccurrence(image, delta, delta_prime):
    # Compute the cross-band co-occurrence matrix for a pair of channels
    num_bins = 256
    R, G, B = cv2.split(image)
    cooccurrence_matrix = np.zeros((num_bins, num_bins), dtype=np.int32)
    for x in range(R.shape[0]):
        for y in range(R.shape[1]):
            i = R[x, y]
            if x + delta[0] < R.shape[0] and y + delta[1] < R.shape[1]:
                j = G[x + delta[0], y + delta[1]]
                cooccurrence_matrix[int(i), int(j)] += 1
    return cooccurrence_matrix

def compute_cooccurrence_matrices(image):
    # Load an image and compute all six co-occurrence matrices
    R, G, B = cv2.split(image)

        
    # Having computed the 6 co-occurrence matrices and stored them in the following variables
    C_delta_R = compute_spatial_cooccurrence(R, (1, 1))
    C_delta_G = compute_spatial_cooccurrence(G, (0, 1))
    C_delta_B = compute_spatial_cooccurrence(B, (1, 0))
    C_delta_RG = compute_crossband_cooccurrence(image, (0, 0), (0, 1))
    C_delta_RB = compute_crossband_cooccurrence(image, (0, 1), (1, 1))
    C_delta_GB = compute_crossband_cooccurrence(image, (0, 0), (1, 0))

    # Create the tensor of shape (256, 256, 3) by stacking the matrices along the last axis
    tensor = np.dstack([C_delta_RG, C_delta_RB, C_delta_GB])
    tensor = tensor.reshape((1, 256, 256, 3))
    return tensor 


img_width, img_height = 256, 256
batch_size = 256

    
def preprocess_image(image):

    return compute_cooccurrence_matrices(image)


image_generator = ImageDataGenerator(validation_split=0.25, preprocessing_function=preprocess_image) 


train_flow = image_generator.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    subset = "training",
    shuffle= True,
    batch_size=batch_size,
    class_mode='binary', 
   
)

valid_flow = image_generator.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    subset = "validation",
    shuffle= True,
    batch_size=batch_size,
    class_mode='binary', 
   
)


def build_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model



model = build_model((256, 256, 3))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

epochs = 20
steps_per_epoch = len(train_flow)

from matplotlib import pyplot as plt


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint( r'C:\Users\harin\OneDrive\Documents\Archana\Indiana\Academics\2_Semester\CV\Project', monitor='val_accuracy', save_best_only=True)
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

plt.plot(range(1, epochs + 1), history.history['loss'])
plt.plot(range(1, epochs + 1), history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_loss.png')

save_model(model, "latest_fft.h5")
