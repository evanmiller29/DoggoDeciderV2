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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop

from sklearn.model_selection import train_test_split

##################################################
## Setting the file paths up and read in labels ##
##################################################

data_path = "C:/Users/evanm/Documents/Data/DoggoDecider/all"
image_path = os.path.join(data_path, 'train', 'train')
train_data_dir = os.path.join(image_path, 'train')
valid_data_dir = os.path.join(image_path, 'validation')

labels = pd.read_csv(os.path.join(data_path, 'labels.csv'))
labels['images'] = labels['id'].apply(lambda x: x + '.jpg')

targets_series = labels['breed']
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)

num_class = one_hot_labels.shape[1]
images_train, images_validation = train_test_split(labels['images'], test_size=0.2, random_state=1234)

labels['train'] = labels['images'].isin(images_train)

#############################################
### Moving the files to train/validation ####
#############################################

breeds = labels['breed'].unique()

for breed in breeds:

    breed_dir = os.path.join(train_data_dir, breed)
    os.makedirs(breed_dir)

    images = labels.loc[(labels['breed'] == breed) & (labels['train'] == True), 'images']

    for image in tqdm(images):

        copyfile(os.path.join(image_path, image), os.path.join(breed_dir, image))

breeds = labels['breed'].unique()

for breed in breeds:

    breed_dir = os.path.join(valid_data_dir, breed)
    os.makedirs(breed_dir)

    images = labels.loc[(labels['breed'] == breed) & (labels['train'] == False), 'images']

    for image in tqdm(images):

        copyfile(os.path.join(image_path, image), os.path.join(breed_dir, image))

num_train = labels.loc[labels['train'] == True, 'images'].nunique()
num_valid = labels.loc[labels['train'] == False, 'images'].nunique()

im_size = 224
batch_size = 32
nb_epoch = 60

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
save_best_model = ModelCheckpoint(filepath='model_.{epoch:02d}_{val_loss:.2f}.hdf5', verbose=1,
        monitor='val_loss')

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#############################################
############# Image generators ##############
#############################################

train_datagen = ImageDataGenerator(rescale= 1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(im_size, im_size),
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical'
    )

validation_generator = validation_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(im_size, im_size),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
    )

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

opt = rmsprop()

model.compile(loss='categorical_crossentropy',
             optimizer = opt,
             metrics = ['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

history = model.fit_generator(train_generator,
                    steps_per_epoch=(num_train // batch_size),
                    epochs=nb_epoch,
                    validation_data=validation_generator,
                    callbacks=[early_stopping, save_best_model],
                    validation_steps=(num_valid// batch_size)
                   )

# Save model
model.save_weights('full_model_weights.h5')
model.save('model.h5')