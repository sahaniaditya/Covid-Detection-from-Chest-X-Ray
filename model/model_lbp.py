"""# LBP(Local Binary Pattern)"""

## LBP IMPLEMENATATION

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_pixel(img, center, x, y):

	new_value = 0

	try:
		# If local neighbourhood pixel
		# value is greater than or equal
		# to center pixel values then
		# set it to 1
		if img[x][y] >= center:
			new_value = 1

	except:
		# Exception is required when
		# neighbourhood value of a center
		# pixel value is null i.e. values
		# present at boundaries.
		pass

	return new_value

# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):

	center = img[x][y]

	val_ar = []

	# top_left
	val_ar.append(get_pixel(img, center, x-1, y-1))

	# top
	val_ar.append(get_pixel(img, center, x-1, y))

	# top_right
	val_ar.append(get_pixel(img, center, x-1, y + 1))

	# right
	val_ar.append(get_pixel(img, center, x, y + 1))

	# bottom_right
	val_ar.append(get_pixel(img, center, x + 1, y + 1))

	# bottom
	val_ar.append(get_pixel(img, center, x + 1, y))

	# bottom_left
	val_ar.append(get_pixel(img, center, x + 1, y-1))

	# left
	val_ar.append(get_pixel(img, center, x, y-1))

	# Now, we need to convert binary
	# values to decimal
	power_val = [1, 2, 4, 8, 16, 32, 64, 128]

	val = 0

	for i in range(len(val_ar)):
		val += val_ar[i] * power_val[i]

	return val

path = '/content/download.jpg'
img_bgr = cv2.imread(path, 1)

height, width, _ = img_bgr.shape

# We need to convert RGB image
# into gray one because gray
# image has one channel only.
img_gray = cv2.cvtColor(img_bgr,
						cv2.COLOR_BGR2GRAY)

# Create a numpy array as
# the same height and width
# of RGB image
img_lbp = np.zeros((height, width),
				np.uint8)

for i in range(0, height):
	for j in range(0, width):
		img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

plt.imshow(img_bgr)
plt.show()

plt.imshow(img_lbp, cmap ="gray")
plt.show()

img_lbp.shape

import numpy as np
def lbp_histogram(filepath, num_bins=256):

     img_bgr = cv2.imread(filepath, 1)
    #  print(img_bgr)
     if(img_bgr is  None):
      return None
     height, width, _ = img_bgr.shape

     # We need to convert RGB image
     # into gray one because gray
     # image has one channel only.
     img_gray = cv2.cvtColor(img_bgr,
						cv2.COLOR_BGR2GRAY)

     # Create a numpy array as
     # the same height and width
     # of RGB image
     img_lbp = np.zeros((height, width),
     				np.uint8)

     for i in range(0, height):
    	 for j in range(0, width):
    		 img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

     histogram ,_ = np.histogram(img_lbp, bins=np.arange(num_bins + 1), density=True)
     return histogram

label_dict = {
    "COVID-POSITIVE" : 1,
    "COVID-NEGATIVE" : 0
}

X_train = []
y_train = []

new_data = data.iloc[300 : 600, :]

new_data.info()

new_data["Label"].value_counts()

new_data.shape

for index, row in new_data.iterrows():
  ans = lbp_histogram(row[0])
  if ans is not None :
    X_train.append(ans)
    y_train.append(label_dict[row[1]])

X_train_array = np.array(X_train)
y_train_array = np.array(y_train)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train_array, y_train_array, test_size=0.2, random_state=42)

## implementing the ANN model

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization , Dropout

model = Sequential()

model.add(Dense(32, activation="relu", input_dim=(256)))
model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Create an SVM model
svm_model = SVC(kernel='rbf', random_state=42)

# Train the SVM model on the training data
svm_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Model Accuracy: {accuracy}')

