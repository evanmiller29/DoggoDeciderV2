import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data_path = "C:/Users/evanm/Documents/Data/DoggoDecider/all"
labels = pd.read_csv(os.path.join(data_path, 'labels.csv'))
labels['images'] = labels['id'].apply(lambda x: x + '.jpg')

targets_series = labels['breed']
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)

X = []
y = []

images = labels['images'].values
labels = labels['breed'].values

i = 0
im_size = 90

for image in tqdm(images):

    img = cv2.imread(os.path.join(data_path, 'train', 'train', image))
    img = cv2.resize(img, (im_size, im_size))

    label = one_hot_labels[i]

    X.append(img)
    y.append(label)

    i += 1

y_raw = np.array(y, np.uint8)
X_raw = np.array(X, np.float32) / 255.

print(y_raw.shape)
print(X_raw.shape)
num_class = y_raw.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=1)

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

model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), verbose=1)
model.save('doggo_decider.h5')
del model  # deletes the existing model
