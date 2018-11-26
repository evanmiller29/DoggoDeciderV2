import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from shutil import copyfile

import keras
from keras.applications.vgg19 import VGG19

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils

from PIL import Image
from sklearn.model_selection import train_test_split

data_path = "C:/Users/evanm/Documents/Data/DoggoDecider/all"
image_path = os.path.join(data_path, 'train', 'train')

labels = pd.read_csv(os.path.join(data_path, 'labels.csv'))
labels['images'] = labels['id'].apply(lambda x: x + '.jpg')

images_train, images_validation = train_test_split(labels['images'], test_size=0.2, random_state=1234)

#############################################
### Moving the files to train/validation ####
#############################################

for image_train in tqdm(images_train):

    copyfile(os.path.join(image_path, image_train), os.path.join(image_path, 'train', image_train))

for image_valid in tqdm(images_validation):

    copyfile(os.path.join(image_path, image_valid), os.path.join(image_path, 'validation', image_valid))

targets_series = labels['breed']
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)

X = []
y = []

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    return image

images = labels['images'].values
labels = labels['breed'].values

i = 0
im_size = 224

for image in tqdm(images):

    img = Image.open(os.path.join(data_path, 'train', 'train', image))
    img = prepare_image(img, target=(im_size, im_size))

    label = one_hot_labels[i]

    X.append(img)
    y.append(label)

    i += 1

y_raw = np.array(y, np.uint8)
X_raw = np.array(X, np.float32) / 255.

print(y_raw.shape)
print(X_raw.shape)
num_class = y_raw.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=1)

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

epochs = 50
batch = 64

datagen.fit(X_train)

#### The above step is creating a memory error. Might want to change to a data_flow_from_directory call

model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
                    steps_per_epoch=len(X_train) / 64, epochs=epochs)

# model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), verbose=0)
model.save('doggo_decider.h5')

del model  # deletes the existing model
