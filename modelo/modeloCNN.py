<<<<<<< HEAD

!pip install keras.applications

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
import numpy as np

size_img=150
batch_size = 20

conv_base = VGG16(weights='imagenet', # Load weights pre-trained on ImageNet.
                  include_top=False,  # Do not include the ImageNet classifier at the top.
                  input_shape=(size_img, size_img, 3))

#resumen de la red
conv_base.summary()

#imagenes
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/content/drive/MyDrive/DI-ML/dataset'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#print('total training images covid:', len(os.listdir(os.path.join(train_dir, 'covid'))))
#print('total training images normal:', len(os.listdir(os.path.join(train_dir, 'normal'))))
#print('total validation images covid:', len(os.listdir(os.path.join(validation_dir, 'covid'))))
#print('total validation images normal :', len(os.listdir(os.path.join(validation_dir, 'normal'))))
#print('total test images covid:', len(os.listdir(os.path.join(test_dir, 'covid'))))
#print('total test images normal:', len(os.listdir(os.path.join(test_dir, 'normal'))))

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized 
        target_size=(size_img, size_img),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(size_img, size_img),
        batch_size=batch_size,
        class_mode='binary')

from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer="Adam",
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()


epochs = 15


history = model.fit(
    train_generator,
	steps_per_epoch=50,
    epochs=epochs, 
    validation_data=validation_generator
    )

model.save('/content/drive/MyDrive/DI-ML/dataset/covid_19_adam.h5')

import matplotlib.pyplot as plt
#print(history.history)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(size_img, size_img),
        batch_size=batch_size,
        class_mode='binary')

##Evaluacion del modelo
results = model.evaluate(test_generator, batch_size=batch_size)

=======

!pip install keras.applications

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
import numpy as np

size_img=150
batch_size = 20

conv_base = VGG16(weights='imagenet', # Load weights pre-trained on ImageNet.
                  include_top=False,  # Do not include the ImageNet classifier at the top.
                  input_shape=(size_img, size_img, 3))

#resumen de la red
conv_base.summary()

#imagenes
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/content/drive/MyDrive/DI-ML/dataset'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#print('total training images covid:', len(os.listdir(os.path.join(train_dir, 'covid'))))
#print('total training images normal:', len(os.listdir(os.path.join(train_dir, 'normal'))))
#print('total validation images covid:', len(os.listdir(os.path.join(validation_dir, 'covid'))))
#print('total validation images normal :', len(os.listdir(os.path.join(validation_dir, 'normal'))))
#print('total test images covid:', len(os.listdir(os.path.join(test_dir, 'covid'))))
#print('total test images normal:', len(os.listdir(os.path.join(test_dir, 'normal'))))

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized 
        target_size=(size_img, size_img),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(size_img, size_img),
        batch_size=batch_size,
        class_mode='binary')

from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer="Adam",
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()


epochs = 15


history = model.fit(
    train_generator,
	steps_per_epoch=50,
    epochs=epochs, 
    validation_data=validation_generator
    )

model.save('/content/drive/MyDrive/DI-ML/dataset/covid_19_adam.h5')

import matplotlib.pyplot as plt
#print(history.history)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(size_img, size_img),
        batch_size=batch_size,
        class_mode='binary')

##Evaluacion del modelo
results = model.evaluate(test_generator, batch_size=batch_size)

>>>>>>> 073c7045ec13629b7af9755f4f82e1ad6af057ec
print("test loss, test acc:", results)