# Covid-Detection-from-Chest-X-Ray

This project deals with the covid-19 detection from chest X-ray using various methods and a comparitive study between them.

### Colab File : https://colab.research.google.com/drive/1LWDPv8UWCeCEcH-9LFtXzwVO5HkJrEe6#scrollTo=wFKyrDpVd2DI
### Resources : 
- [HOG FEATURES](https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#:~:text=HOG%2C%20or%20Histogram%20of%20Oriented,vision%20tasks%20for%20object%20detection).
- [OPEN CV](https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html)
- [HOG](https://medium.com/@girishajmera/hog-histogram-of-oriented-gradients-an-amazing-feature-extraction-engine-for-medical-images-5a2203b47ccd#:~:text=HOG%20is%20a%20feature%20descriptor,local%20contrast%20in%20overlapping%20blocks.)
- [HOG BASICS](https://medium.com/analytics-vidhya/a-gentle-introduction-into-the-histogram-of-oriented-gradients-fdee9ed8f2aa)
- [LBP](https://towardsdatascience.com/the-power-of-local-binary-patterns-3134178af1c7)


This repository contains code for detecting COVID-19 from chest X-ray images using three different techniques: Histogram of Oriented Gradients (HOG), Convolutional Neural Network (CNN), and Local Binary Pattern (LBP).

## Introduction

The outbreak of COVID-19 has posed a significant challenge worldwide, and the early and accurate detection of the disease is crucial for effective management and control. Chest X-ray imaging has emerged as a valuable tool for diagnosing COVID-19 due to its wide availability and rapid turnaround time. In this project, we explore three different techniques for automated COVID-19 detection from chest X-ray images.The quality of the chest-X-ray images is not good , so a lot of preprocessing is required .

## Techniques Used

1. **Histogram of Oriented Gradients (HOG)**:
    - HOG is a feature descriptor widely used in object detection and image classification tasks.
    - We extract HOG features from chest X-ray images and feed them into a machine learning model for COVID-19 detection.

2. **Convolutional Neural Network (CNN)**:
    - CNNs are deep learning models known for their effectiveness in image classification tasks.
    - We train a CNN model on a dataset of chest X-ray images to learn features and classify them into COVID-19 positive or negative.

3. **Local Binary Pattern (LBP)**:
    - LBP is a texture descriptor used for texture classification and face recognition.
    - We extract LBP features from chest X-ray images and use them to train a machine learning model for COVID-19 detection.

## Repository Structure

- `implementation`: Contains from scratch implementations of lbp and hog.
- `model`: Contains different model implementation.
- `Covid_Detection_Using_X_Ray.ipynb/`: Contains Python scripts for implementing HOG, CNN, and LBP techniques for COVID-19 detection.
- `README.md`: This file, providing an overview of the project.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/sahaniaditya/Covid-Detection-from-Chest-X-Ray.git
    git clone https://github.com/shikhar5647/Covid-Detection-from-Chest-X-Ray.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Covid-Detection-from-Chest-X-Ray
    ```

3. Follow the instructions in the respective directories (`code/`) to run the code for each technique.

## Results


- HOG feature extraction technique and the implementation of various ML models achieved the best accuracy of 74.56%.
- CNN technique achieved an accuracy of 91.66%. Both Tensorflow and Pytorch implementation were done and the results demonstrated.
- LBP technique achieved an accuracy of 95.66%. A simple neural network was implemented after the extraction of the features.


## Conclusion

In this project, we explored three different techniques for COVID-19 detection from chest X-ray images. Each technique has its advantages and limitations. Further research and experimentation could lead to improved models for more accurate and reliable COVID-19 detection.

## Contributors

- [Aditya Sahani](https://github.com/sahaniaditya)
- [Shikhar Dave](https://github.com/shikhar5647)
         
