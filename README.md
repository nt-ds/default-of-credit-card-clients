# default-of-credit-card-clients

Our purpose is to build a model that predicts whether or not a person is going to default on their credit card in the next month, with the goal of identifying and assisting them before they default.

## Data

Data was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). It contains information on the payment habits of 30,000 credit card holders in Taiwan from 2005.

The data is conveniently provided in the data folder of this repository. Multiple forms of the data are provided. The raw data is in the raw subfolder, and the processed data is in the processed subfolder. There are 2 processed data sets, one which is unscaled and the other in which numerical (non-dummy) variables are scaled between 0 and 1. 

## Notebooks

The full project is comprised of four notebooks. The first notebook, 1.0-eda, contains the process used to explore and clean the data. The second and third notebooks, titled 2.0-models and 3.0-models-transformed-data, each contain several models for the data. The fourth notebook, Credit-Card-Default, provides an overview and exploration of the best performing model.

## Results
We attempted to fit models using decision trees, random forest, xgboost, logistic regression, and SVM. Because our classes were imbalanced, we also try models on undersampled, oversampled, and SMOTE created data. Decision trees yielded the best models, as determined by recall. They were most successful on undersampled data. Our decision tree model revealed that being 2 months late on payment and bill amount are strong predictors of whether a credit card user will default. 
