# CVDPrediction - Use Machine Learning with Longitudinal EHR and Genetic Data to Improve Cardiovascular Prediction

This is the source code of paper
Evaluating and Mitigating Bias in ML Models for CVD.docx


The project explores several machine learning and deep learning models on features extracted from EHR for Cardiovascular disease prediction.

Models include:
* logistic regression, 
* random forests, 
* gradient boosting trees, 
* convolutional neural networks (CNN) 
* recurrent neural networks with long short-term memory (LSTM) units.

## How to run the code

(1) Put your own dataset under the data folder. In our paper, we compared two different feature extraction: a) using aggregated values for each feature and b) using temporal values (yearly) for each feature.
We did not upload the raw data due to the privacy concern. We only upload the headers of the data that shows the feature structures.

data.csv should be the input data contains features and class labels. X.npy is the features, Y. npy is the labels.

(2) The jupyter notebooks showed the result using different machine learning and LSTM models. 

(3) ./src contains several python files that implement several models and cross validations

 src/classification/run_benchamrk.py runs the nested cross validations to compare the different machine learning models.
 
 src/tune_DNN.py, src/tune_CNN.py and src/tune_LSTM to run the cross validations on deep learning models.
 
