""""
CNN IMPLEMENTATION ON COVID DATA USING TENSORFLOW

"""

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

model = Sequential()

model.add(Conv2D(4, kernel_size=(3,3), padding="valid", activation="relu", input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides=2, padding="valid"))

model.add(Conv2D(8, kernel_size=(3,3), padding="valid", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides=2, padding="valid"))

model.add(Conv2D(16, kernel_size=(3,3), padding="valid", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides=2, padding="valid"))

model.add(Flatten())

model.add(Dense(16, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.1))


model.add(Dense(1, activation="sigmoid"))

model.summary()

train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/drive/MyDrive/X-ray/train',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
)

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label

train_ds = train_ds.map(process)

x_train_lst = []
y_train_lst = []

for image, label in train_ds:
  # print(image, "image shape :  ",image.shape,  "Lable :  ", label)
  x_train_lst.append(image)
  y_train_lst.append(label)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds, epochs=1, batch_size=32)

import cv2

import matplotlib.pyplot as plt

test_img = cv2.imread("/content/drive/MyDrive/X-ray/COVID/16654_4_1.jpg")

plt.imshow(test_img)

test_img = cv2.resize(test_img,(256,256))

test_input = test_img.reshape((1,256,256,3))

model.predict(test_input)