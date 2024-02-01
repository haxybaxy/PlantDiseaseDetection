# Plant Disease Detection Using Machine Learning

This project addresses the challenge of detecting plant diseases using technology. The goal is to build a machine learning model that can accurately classify images of apple leaves as healthy or diseased.
This was made for learning purposes as I believe it is a good application of classic ML skills, and by no means is the best way to tackle this problem.

## Dataset Overview

The dataset, sourced from the Plant-Village-Dataset (available at [GitHub repository](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)), consists of color images of apple leaves. It includes two folders: 'Diseased' (with leaves affected by Apple Scab, Black Rot, or Cedar Apple Rust) and 'Healthy' (with green, healthy leaves).

## Image Properties

- File Type: JPG
- Size: 256x256 pixels
- Resolution: 96 dpi (both horizontal and vertical)
- Bit Depth: 24

## Methodology

The preprocessing includes loading images, converting them from BGR to RGB (and then to HSV for better color and intensity separation), and segmenting to extract leaf colors. Features are extracted using color statistics (Color Histogram), shape (Hu Moments), and texture (Haralick Texture), and then stacked with numpy.

The dataset is split into 80% training and 20% testing sets, with feature scaling using Min-Max Scaler. Features are saved in an HDF5 file format.

Multiple machine learning models were trained and validated using 10-fold cross-validation. The models include Logistic Regression, LDA, KNN, Decision Trees, Random Forest, Naive Bayes, and SVM.

The best-performing model, the Random Forest Classifier, achieved a 97% accuracy rate on the test set.

## Additional Resources

- Utils: Python file for converting image labels in the training folders.
- Image Classification: Training dataset and the Jupyter notebook for Plant Disease Detection.
- Testing Notebook: Detailed specification of functions applied to the leaf images.