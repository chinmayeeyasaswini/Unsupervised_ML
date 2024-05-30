# Unsupervised Machine Learning Models

This repository contains implementations of two unsupervised machine learning algorithms, K-Means clustering and K-Nearest Neighbors (KNN), each applied to different datasets. The K-Means model is trained on the Titanic dataset, and the KNN model is trained on the Breast Cancer Wisconsin dataset. The necessary data preprocessing steps are included within the scripts.

## Repository Structure

- KMeans.py: Contains the implementation of the K-Means clustering algorithm along with preprocessing steps for the Titanic dataset.
- KNearestNeighbour.py: Contains the implementation of the K-Nearest Neighbors algorithm along with preprocessing steps for the Breast Cancer Wisconsin dataset.

## K-Means Clustering on Titanic Dataset

### Data Preprocessing and Model Training
The KMeans.py script performs the following steps:
- Loads and preprocesses the Titanic dataset:
  - Handles missing values.
  - Encodes categorical variables.
  - Normalizes numerical features.
- Applies the K-Means algorithm to cluster passengers.
- Outputs the clustering results.

### Usage
To run the K-Means clustering on the Titanic dataset:
bash
python KMeans.py


## K-Nearest Neighbors on Breast Cancer Wisconsin Dataset

### Data Preprocessing and Model Training
The KNearestNeighbour.py script performs the following steps:
- Loads and preprocesses the Breast Cancer Wisconsin dataset:
  - Handles missing values.
  - Encodes categorical variables.
  - Normalizes numerical features.
- Applies the KNN algorithm to classify tumors.
- Outputs the classification results.

### Usage
To run the K-Nearest Neighbors classification on the Breast Cancer Wisconsin dataset:
bash
python KNearestNeighbour.py


## Dependencies
Make sure you have the following packages installed:
- pandas
- numpy
- scikit-learn

You can install the dependencies using the following command:
bash
pip install pandas numpy scikit-learn


## Contributing
Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.