"""## USING HOG(HISTOGRAM OF GRADIENTS)"""

import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

# Function to compute HOG features for an image
def compute_hog_features(image):
    # Compute HOG features
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)


    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return features, hog_image

# List to store computed HOG features and HOG images
hog_features_list = []
hog_images_list = []

# Process and compute HOG features for all images
for image_path in images:
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)

    # Compute HOG features and HOG image
    features, hog_image = compute_hog_features(image)

    # Append features and HOG image to the lists
    hog_features_list.append(features)
    hog_images_list.append(hog_image)

# Convert the lists to NumPy arrays
hog_features_array = np.array(hog_features_list)
hog_images_array = np.array(hog_images_list)

# Display a sample HOG image
sample_hog_image = hog_images_array[0]
plt.imshow(sample_hog_image, cmap='gray')
plt.title('Sample HOG Image')
plt.axis('off')
plt.show()

print(len(labels))  # Print the length of the labels list
print(labels)       # Print the entire labels list to examine its content



labels_array = np.array(labels)
print(labels[900])

import matplotlib.pyplot as plt

# Assuming hog_features_array is a 1D array
hog_features = hog_features_array

plt.figure(figsize=(8, 6))
plt.hist(hog_features, bins=20)  # Adjust the number of bins as needed
plt.title('Histogram of HOG Features')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid()
plt.show()

print("hog_features_array shape:", hog_features_array.shape)
print("labels_array shape:", labels_array.shape)

# Check data types
print("hog_features_array dtype:", hog_features_array.dtype)
print("labels_array dtype:", labels_array.dtype)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hog_features_array, labels_array[:1638], test_size=0.2, random_state=42)

# unique_classes_train = np.unique(y_train)
# print("Unique Classes in Training Data:", unique_classes_train)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Create an SVM model
svm_model = SVC(kernel='rbf', random_state=42)

# Train the SVM model on the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Model Accuracy: {accuracy}')

import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming X_train, X_test, y_train, y_test are already defined with shape (samples, height, width, channels)

# Reshape the data to (samples, height * width * channels)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Model: {name}')
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{classification_rep}')
    print('---------------------')
