This is the source code of paper
Evaluating and Mitigating Bias in Machine Learning Models for Cardiovascular Disease Prediction


The project explores several machine learning and deep learning models on features extracted from EHR for Cardiovascular disease prediction.

Models include:
* AHA pooled cohort risk equations
* logistic regression, 
* random forests, 
* gradient boosting trees, 
* recurrent neural networks with long short-term memory (LSTM) units.

## Descriptions of the code

(1) Put your own dataset under the data folder. We did not upload the raw data due to the privacy concern. Please contact us if you have any concerns.

(2) The fairness evaluations are in the "packaged fairness tests" folder. Each file is for a specific task, while they share a common framework. 

(3) The deep learning folder includes files to train the LSTM model, to evaluate, and to mitigate bias. The results are also in the folder.

(3) The "result analysis" folder includes files that turns the .csv results into summary statistics. 

(4) The "Statistical analysis" folder includes all statistical analysis. 

(5) "Models" folder include all models we used for hyper-parameter tuning. The model with the best performance is used for external validation.

(6) ./src contains several python files that implement several models and cross validations
 
