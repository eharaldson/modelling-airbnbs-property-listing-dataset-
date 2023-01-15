from tabular_data import load_airbnb
from pytorch_datasets import AirbnbNightlyPriceImageDataset

from sklearn import linear_model, model_selection, metrics, preprocessing, tree, ensemble, svm, neighbors, gaussian_process
from torch.utils.data import random_split
from itertools import product

import torch.utils.data
import torch
import torch.nn.functional as F
import torch.utils.tensorboard

import os
import json
import yaml
import time
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(20)

''' Regression '''
# Completes cross validation of a model on different parameters and returns the best model
def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, parameter_dictionary):
    '''
    Example usage:
    p_dict = {'alpha': [10,1,0.1,0.01],
            'max_iter': [100, 200, 300, 400]}

    data = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test}
    model, hyperparamters, performance_metrics = custom_tune_regression_model_hyperparameters(model_class=linear_model.Ridge, **data, parameter_dictionary=p_dict)
    '''
    hyperparameter_combinations = [dict(zip(parameter_dictionary, v)) for v in product(*parameter_dictionary.values())]

    best_rmse = np.inf

    for hyperparameter_combination in hyperparameter_combinations:

        model = model_class(**hyperparameter_combination)
        model.fit(X_train, y_train)

        y_hat = model.predict(X_val)
        rmse = metrics.mean_squared_error(y_val, y_hat, squared=False)

        print(model, rmse)

        if rmse < best_rmse:

            best_model = model
            best_rmse = rmse
            best_hyperparamater_values = hyperparameter_combination
            
            validation_r2_score = model.score(X_val, y_val)

            y_hat = model.predict(X_test)
            test_rmse = metrics.mean_squared_error(y_test, y_hat, squared=False)
            test_r2_score = model.score(X_test, y_test)

            best_performance_metrics = {'validation_RMSE': rmse,
                                        'validation_R2': validation_r2_score,
                                        'test_RMSE': test_rmse,
                                        'test_R2': test_r2_score}

    return best_model, best_hyperparamater_values, best_performance_metrics

# Uses sklearn's GridSearchCV on a linear regression model to find the best model
def tune_regression_model_hyperparameters(estimator, X_train, X_test, y_train, y_test, hyperparameters):

    clf = model_selection.GridSearchCV(estimator=estimator,
                                       scoring='neg_root_mean_squared_error',
                                       param_grid=hyperparameters,
                                       n_jobs=-1)

    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    model_hyperparameters = clf.best_params_
    validation_rmse_score = -clf.best_score_

    y_hat = clf.predict(X_test)
    test_rmse_score = -clf.score(X_test, y_test)
    test_r2_score = metrics.r2_score(y_test, y_hat)

    if 'estimator' in model_hyperparameters.keys():
        model_hyperparameters['estimator'] = str(model_hyperparameters['estimator'])

    best_score_metrics = {'validation_RMSE': validation_rmse_score, 'test_RMSE': test_rmse_score, 'test_R2': test_r2_score}

    return best_model, model_hyperparameters, best_score_metrics

# Function for running GridSearchCV for Adaboost
def adaboost_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'estimator': [tree.DecisionTreeRegressor(), svm.NuSVR(), linear_model.SGDRegressor(), neighbors.KNeighborsRegressor()],
                   'n_estimators': [5, 20, 40, 80, 150, 250]}
    estimator = ensemble.AdaBoostRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/adaboost', model, model_hyperparameters, model_score_metrics)

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for gradient boosting
def gradientboost_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'n_estimators': [5, 20, 40, 80, 150, 250],
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'learning_rate': [0.5, 1, 5, 10]}
    estimator = ensemble.GradientBoostingRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/gradient_boosting', model, model_hyperparameters, model_score_metrics)

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for random forest
def random_forest_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    'n_estimators': [5, 20, 50, 100, 200],
                    'max_depth': [1, 5, 25, 50, 100, 200],
                    'max_features' : ["sqrt", "log2", None], 
                    'min_samples_leaf': [1, 2, 3, 4, 5]}
    estimator = ensemble.RandomForestRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/random_forest', model, model_hyperparameters, model_score_metrics)

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for decision tree regressor
def decision_tree_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    'max_depth': [1, 5, 25, 50, 100, 200]}
    estimator = tree.DecisionTreeRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/decision_tree', model, model_hyperparameters, model_score_metrics)

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for sgd regressor
def sgd_regressor_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'penalty': ['l1', 'l2'],
                        'alpha': [10, 1, 0.1, 0.001, 0.0001, 0.00001, 0],
                        'max_iter': [1000, 10000, 100000, 1000000]}
    estimators = linear_model.SGDRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/linear_regression', model, model_hyperparameters, model_score_metrics)    

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for linear SVM regressor
def linear_SVM_regressor_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'epsilon': [0, 0.5, 1, 5, 10],
                       'C': [0.25, 0.5, 1, 1.5, 2],
                       'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                       'max_iter': [500, 1000, 1500, 2000]}
    estimators = svm.LinearSVR()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/linear_svm', model, model_hyperparameters, model_score_metrics)    

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for SVR regressor
def SVM_regressor_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                       'degree': [1, 2, 3, 4, 5],
                       'gamma': ['auto', 'scale', 0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'C': [0.25, 0.5, 1, 1.5, 2],
                       'epsilon': [0, 0.5, 1, 5, 10]}
    estimators = svm.SVR()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/svm', model, model_hyperparameters, model_score_metrics)    

    return model, model_hyperparameters, model_score_metrics


''' Classification '''
# Return important classification metrics for a model
def classification_metrics(model, X_train, X_test, y_train, y_test, multiclass = False, multiclass_average = 'macro'):

    y_train_prediction = model.predict(X_train)
    y_test_prediction = model.predict(X_test)

    if multiclass == False:
        average = 'binary'
    else:
        average = multiclass_average

    f1_score_train = metrics.f1_score(y_train, y_train_prediction, average=average)
    precision_train = metrics.precision_score(y_train, y_train_prediction, average=average)
    recall_train = metrics.recall_score(y_train, y_train_prediction, average=average)
    accuracy_train = metrics.accuracy_score(y_train, y_train_prediction)

    f1_score_test = metrics.f1_score(y_test, y_test_prediction, average=average)
    precision_test = metrics.precision_score(y_test, y_test_prediction, average=average)
    recall_test = metrics.recall_score(y_test, y_test_prediction, average=average)
    accuracy_test = metrics.accuracy_score(y_test, y_test_prediction)

    return {'train': {'f1': f1_score_train,
                      'precision': precision_train,
                      'recall': recall_train,
                      'accuracy': accuracy_train
                      },
            'test': {'f1': f1_score_test,
                     'precision': precision_test,
                     'recall': recall_test,
                     'accuracy': accuracy_test}
            }

# Uses sklearn's GridSearchCV on a classification model to find the best model
def tune_classification_model_hyperparameters(estimator, X_train, X_test, y_train, y_test, hyperparameters):

    clf = model_selection.GridSearchCV(estimator=estimator,
                                       scoring='accuracy',
                                       param_grid=hyperparameters,
                                       n_jobs=-1)

    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    model_hyperparameters = clf.best_params_
    validation_accuracy_score = clf.best_score_

    # With ensemble algorithms I will test different estimators but this is not always the case so this if statement handles this
    if 'estimator' in model_hyperparameters.keys():
        model_hyperparameters['estimator'] = str(model_hyperparameters['estimator'])

    # Obtain metrics on the train and test data here
    train_and_test_metrics = classification_metrics(best_model, X_train, X_test, y_train, y_test, multiclass = True)

    best_score_metrics = {'validation_accuracy': validation_accuracy_score, 'train_and_test': train_and_test_metrics}

    return best_model, model_hyperparameters, best_score_metrics

# Function for running GridSearchCV for logistic regression
def logistic_regression_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'penalty': ['l1', 'l2', 'elasticnet', None],
                       'C': [0.25, 0.5, 0.75, 1, 1.5, 2],
                       'max_iter': [50, 100, 500, 1000, 5000]}
    estimators = linear_model.LogisticRegression(multi_class='multinomial')
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/logistic_regression', model, model_hyperparameters, model_score_metrics)    

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for decision tree classifier
def decision_tree_classifier_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'criterion': ['gini', 'entropy', 'log_loss'],
                       'splitter': ['best', 'random'],
                       'max_depth': [1, 5, 25, 50, 100, 200],
                       'min_samples_leaf': [1, 2, 3, 5, 10]}
    estimators = tree.DecisionTreeClassifier()
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/decision_tree', model, model_hyperparameters, model_score_metrics) 

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for random_forest classifier
def random_forest_classifier_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'criterion': ['gini', 'entropy', 'log_loss'],
                       'n_estimators': [5, 20, 50, 100, 200, 400],
                       'max_depth': [1, 5, 25, 50, 100],
                       'min_samples_leaf': [1, 2, 3, 4, 5]}
    estimators = ensemble.RandomForestClassifier()
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/random_forest', model, model_hyperparameters, model_score_metrics) 

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for gradient boosting classifier
def gradientboost_classifier_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'loss': ['log_loss', 'deviance', 'exponential'],
                       'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                       'n_estimators': [5, 20, 50, 100, 200, 400],
                       'max_depth': [1, 3, 25, 50, 100],
                       'min_samples_leaf': [1, 2, 3, 4, 5]}
    estimators = ensemble.GradientBoostingClassifier()
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/gradientboost', model, model_hyperparameters, model_score_metrics) 

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for nearest neighbour classifier
def nearest_neighbour_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'n_neighbors': [1, 2, 3, 5, 7, 10],
                       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                       'leaf_size': [10, 20, 30, 40, 50],
                       'p': [1, 2, 3, 4],
                       'weights': ['uniform', 'distance']}
    estimators = neighbors.KNeighborsClassifier(n_jobs=-1)
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/k_nearest_neighbors', model, model_hyperparameters, model_score_metrics) 

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for linear SVM classifier
def linear_SVM_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'penalty': ['l1', 'l2'],
                       'loss': ['hinge', 'squared_hinge'],
                       'C': [0.25, 0.5, 0.75, 1, 1.5, 2],
                       'multi_class': ['ovr', 'crammer_singer'],
                       'max_iter': [500, 1000, 1500, 2000, 5000]}
    estimators = svm.LinearSVC()
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/linear_svm', model, model_hyperparameters, model_score_metrics) 

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for SVM classifier
def SVM_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'degree': [1, 2, 3, 4, 5],
                       'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                       'C': [0.25, 0.5, 0.75, 1, 1.5, 2],
                       'gamma': ['auto', 'scale', 0.00001, 0.0001, 0.001, 0.01, 0.1]}
    estimators = svm.SVC()
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/svm', model, model_hyperparameters, model_score_metrics) 

    return model, model_hyperparameters, model_score_metrics

# Function for running GridSearchCV for gaussian process classifier
def gaussian_process_CV(X_train, X_test, y_train, y_test, save_ = True):
    hyperparameters = {'max_iter_predict': [50, 100, 200, 400, 1000],
                       'multi_class': ['one_vs_rest', 'one_vs_one']}
    estimators = gaussian_process.GaussianProcessClassifier(n_jobs=-1)
    model, model_hyperparameters, model_score_metrics = tune_classification_model_hyperparameters(estimators, X_train, X_test, y_train, y_test, hyperparameters)

    if save_ == True:
        save_model('models/classification/gaussian_process', model, model_hyperparameters, model_score_metrics) 

    return model, model_hyperparameters, model_score_metrics


''' General '''
# Function to preprocess data and obtain it in a clean format
def generate_processed_data(regression = True):

    if regression == True:
        label_name = 'Price_Night'
    else:
        label_name = 'Category'

    # Load data for classification
    X, y = load_airbnb(label_name=label_name)
    X = np.array(X)
    y = np.array(y).reshape(-1)

    if regression == False:
        # Label encode the Category labels
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

    # Split data into train, validation and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Set up a scaler to standardize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Plot the unique categorical values in a histogram
def plot_categorical_data():
    X, y = load_airbnb('Category')

    plt.figure()
    y['Category'].value_counts().plot(kind='bar')
    plt.xlabel('Listing categories')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Function for running GridSearchCV on all the models
def evaluate_all_models(X_train, X_test, y_train, y_test, task_folder):

    if task_folder == 'models/classification':
        # Run the classification models and save the best one after cross validation
        logistic_regression_CV(X_train, X_test, y_train, y_test)
        decision_tree_classifier_CV(X_train, X_test, y_train, y_test)
        random_forest_classifier_CV(X_train, X_test, y_train, y_test)
        gradientboost_classifier_CV(X_train, X_test, y_train, y_test)
        nearest_neighbour_CV(X_train, X_test, y_train, y_test)
        linear_SVM_CV(X_train, X_test, y_train, y_test)
        SVM_CV(X_train, X_test, y_train, y_test)
        gaussian_process_CV(X_train, X_test, y_train, y_test)

    elif task_folder == 'models/regression':
        # Run the regression models and save the best one after cross validation
        sgd_regressor_CV(X_train, X_test, y_train, y_test)
        decision_tree_CV(X_train, X_test, y_train, y_test)
        random_forest_CV(X_train, X_test, y_train, y_test)
        gradientboost_CV(X_train, X_test, y_train, y_test)
        adaboost_CV(X_train, X_test, y_train, y_test)
        linear_SVM_regressor_CV(X_train, X_test, y_train, y_test)
        SVM_regressor_CV(X_train, X_test, y_train, y_test)  

# Find the best model and return it
def find_best_model(task_folder):

    cwd = os.getcwd()

    if task_folder == 'models/regression':
        folder = os.path.join(cwd, 'models/regression')
        subfolders = ['adaboost', 'decision_tree', 'gradient_boosting', 'linear_regression', 'random_forest', 'linear_svm', 'svm']

        validation_score = "validation_RMSE"
        best_score = 99999999999
        model_type = 'r'
    elif task_folder == 'models/classification':
        folder = os.path.join(cwd, 'models/classification')
        subfolders = ['gaussian_process', 'decision_tree', 'gradientboost', 'logistic_regression', 'random_forest', 'k_nearest_neighbors', 'linear_svm', 'svm']

        validation_score = "validation_accuracy"
        best_score = 0
        model_type = 'c'

    for subfolder in subfolders:
        beats_others = False
        sub_directory = os.path.join(folder, subfolder)
        metrics_path = os.path.join(sub_directory, 'metrics.json')

        with open(metrics_path, 'rb') as f:
            metrics = json.load(f)

        if model_type == 'r':
            if metrics[validation_score] < best_score:
                beats_others = True
        elif model_type == 'c':
            if metrics[validation_score] > best_score:
                beats_others = True

        if beats_others == True:
            best_score = metrics[validation_score]
            best_metrics = metrics

            model_path = os.path.join(sub_directory, 'model.joblib')
            with open(model_path, 'rb') as f:
                best_model = pickle.load(f)

            hyperparameter_path = os.path.join(sub_directory, 'hyperparameters.json')
            with open(hyperparameter_path, 'rb') as f:
                best_hyperparameters = json.load(f)

    return best_model, best_hyperparameters, best_metrics

# Function to save model
def save_model(folder, model, model_hyperparameters, model_score_metrics):

    cwd = os.getcwd()

    if 'neural_networks' in folder.split('/'):
        model_filename = os.path.join(cwd, folder+'/model.pt')
        torch.save(model.state_dict(), model_filename)
    else:
        model_filename = os.path.join(cwd, folder+'/model.joblib')
        pickle.dump(model, open(model_filename, 'wb'))

    hyperparameter_filename = os.path.join(cwd, folder+'/hyperparameters.json')
    score_filename = os.path.join(cwd, folder+'/metrics.json')

    # Save hyperparameters
    with open(hyperparameter_filename, 'w') as outfile:
        json.dump(model_hyperparameters, outfile)

    # Save metrics
    with open(score_filename, 'w') as outfile:
        json.dump(model_score_metrics, outfile)


''' Deep Learning '''
# Neural Network model class
class NNRegression(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        layer_widths = config['hidden_layer_width']
        modules = []
        for i in range(config['model_depth']):
            modules.append(torch.nn.Linear(in_features=layer_widths[i][0], out_features=layer_widths[i][1]))
            if i < config['model_depth']-1:
                modules.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*modules)
    
    def forward(self, features):
        return self.layers(features)

# Function to get the instance of the optimiser class from the name
def get_optimiser_from_name(model, name, lr):
    if name == 'Adam':
        return torch.optim.Adam(params=model.parameters(), lr=lr)
    elif name == 'Adadelta':
        return torch.optim.Adadelta(params=model.parameters(), lr=lr)
    elif name == 'Adagrad':
        return torch.optim.Adagrad(params=model.parameters(), lr=lr)
    elif name == 'AdamW':
        return torch.optim.AdamW(params=model.parameters(), lr=lr)
    elif name == 'SparseAdam':
        return torch.optim.SparseAdam(params=model.parameters(), lr=lr)
    elif name == 'Adamax':
        return torch.optim.Adamax(params=model.parameters(), lr=lr)
    elif name == 'ASGD':
        return torch.optim.ASGD(params=model.parameters(), lr=lr)
    elif name == 'LBFGS':
        return torch.optim.LBFGS(params=model.parameters(), lr=lr)
    elif name == 'NAdam':
        return torch.optim.NAdam(params=model.parameters(), lr=lr)
    elif name == 'RAdam':
        return torch.optim.RAdam(params=model.parameters(), lr=lr)
    elif name == 'RMSprop':
        return torch.optim.RMSprop(params=model.parameters(), lr=lr)
    elif name == 'Rprop':
        return torch.optim.Rprop(params=model.parameters(), lr=lr)
    else:
        return torch.optim.SGD(params=model.parameters(), lr=lr)

# Function to train a neural network
def train(model, dataloader, hyperparams, epochs=10):

    optimiser = get_optimiser_from_name(model=model, name=hyperparams['optimiser'], lr=hyperparams['lr'])

    writer = torch.utils.tensorboard.SummaryWriter()

    batch_index = 0

    for epoch in range(epochs):
        for batch in dataloader['train']:
            features, labels = batch
            predictions = model(features)
            loss = F.mse_loss(predictions, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            val_features, val_labels = next(iter(dataloader['validation']))
            val_predictions = model(val_features)
            val_loss = F.mse_loss(val_predictions, val_labels)

            writer.add_scalar(tag='Loss', scalar_value=loss.item(), global_step=batch_index)
            writer.add_scalar(tag='Validation Loss', scalar_value=val_loss.item(), global_step=batch_index)
            batch_index += 1

# Function that reads the nn config yaml file and returns a dictionary
def get_nn_config():
    with open("nn_config.yaml", "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

# Evaluates the neural networks metrics
def evaluate_nn(model, data_loader, nn_config, epochs=10):

    start = time.time()
    train(model, data_loader, hyperparams=nn_config, epochs=epochs)

    print(model.modules())
    training_duration = time.time() - start

    x_train, y_train = next(iter(data_loader['train_metrics']))
    x_validation, y_validation = next(iter(data_loader['validation']))
    x_test, y_test = next(iter(data_loader['test']))

    start = time.time()
    train_predictions = model(x_train)
    validation_predictions = model(x_validation)
    test_predictions = model(x_test)
    inference_latency = (time.time() - start) / (len(x_train) + len(x_validation) + len(x_test))

    train_rmse = metrics.mean_squared_error(y_train.numpy(), train_predictions.detach().numpy(), squared=False)
    train_r2 = metrics.r2_score(y_train.numpy(), train_predictions.detach().numpy())
    validation_rmse = metrics.mean_squared_error(y_validation.numpy(), validation_predictions.detach().numpy(), squared=False)
    validation_r2 = metrics.r2_score(y_validation.numpy(), validation_predictions.detach().numpy())
    test_rmse = metrics.mean_squared_error(y_test.numpy(), test_predictions.detach().numpy(), squared=False)
    test_r2 = metrics.r2_score(y_test.numpy(), test_predictions.detach().numpy())


    metrics_dictionary = {
        'RMSE_loss': {
            'train': float(train_rmse),
            'validation': float(validation_rmse),
            'test': float(test_rmse)
        },
        'R_squared': {
            'train': float(train_r2),
            'validation': float(validation_r2),
            'test': float(test_r2)
        },
        'training_duration': training_duration,
        'inference_latency': inference_latency
    }

    return metrics_dictionary

# Create a folder with the current time in the name to then save a neural network model
def save_nn(model, hyperparams, metrics, regression=True):

    cwd = os.getcwd()
    if regression == True:
        folder_path = os.path.join(cwd, 'models/neural_networks/regression')
    else:
        folder_path = os.path.join(cwd, 'models/neural_networks/classification')

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    folder_path = os.path.join(folder_path, current_time)

    os.mkdir(folder_path)

    save_model(folder=folder_path, model=model, model_hyperparameters=hyperparams, model_score_metrics=metrics)

# Generates 16 different configuration settings of a neural network
def generate_nn_configs():

    configs = []
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 14], [14, 1]],
        'model_depth': 2
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.001,
        'hidden_layer_width': [[11, 14], [14, 1]],
        'model_depth': 2
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 17], [17, 1]],
        'model_depth': 2
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.001,
        'hidden_layer_width': [[11, 17], [17, 1]],
        'model_depth': 2
    })

    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 20], [20, 1]],
        'model_depth': 2
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.001,
        'hidden_layer_width': [[11, 20], [20, 1]],
        'model_depth': 2
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 24], [24, 1]],
        'model_depth': 2
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 24], [24, 1]],
        'model_depth': 2
    })

    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 33], [33, 10], [10, 1]],
        'model_depth': 3
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 33], [33, 5], [5, 1]],
        'model_depth': 3
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 25], [25, 14], [14, 1]],
        'model_depth': 3
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 16], [16, 23], [23, 1]],
        'model_depth': 3
    })

    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 40], [40, 7], [7, 1]],
        'model_depth': 3
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.001,
        'hidden_layer_width': [[11, 35], [35, 14], [14, 1]],
        'model_depth': 3
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.0001,
        'hidden_layer_width': [[11, 22], [22, 11], [11, 1]],
        'model_depth': 3
    })
    configs.append({
        'optimiser': 'SGD',
        'lr': 0.001,
        'hidden_layer_width': [[11, 22], [22, 8], [8, 1]],
        'model_depth': 3
    })

    return configs

# Finds the best neural network model
def find_best_nn():
    configs = generate_nn_configs()

    data = AirbnbNightlyPriceImageDataset()
    train_data, validation_data, test_data = random_split(data, [0.7, 0.15, 0.15])
    batch_size = 64
    data_loaders = {
        'train': torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        ),
        'train_metrics': torch.utils.data.DataLoader(
            train_data,
            batch_size=len(train_data),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        ),
        'validation': torch.utils.data.DataLoader(
            validation_data,
            batch_size=len(validation_data),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        ),
        'test': torch.utils.data.DataLoader(
            test_data,
            batch_size=len(test_data),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
    }

    best_validation_score = np.inf
    for config in configs:
        model = NNRegression(config=config)
        model_metrics = evaluate_nn(model, data_loaders, config, 40)
        save_nn(model, config, model_metrics)

        if model_metrics['RMSE_loss']['validation'] < best_validation_score:
            best_validation_score = model_metrics['RMSE_loss']['validation']
            best_model = model
            best_config = config
            best_metrics = model_metrics

    return best_model, best_config, best_metrics


if __name__ == "__main__":


    best_model, best_config, best_metrics = find_best_nn()

    folder_path = os.path.join(os.getcwd(), 'models/neural_networks/regression/best_model')
    save_model(folder=folder_path, model=best_model, model_hyperparameters=best_config, model_score_metrics=best_metrics)
    
    # nn_config = get_nn_config()
    # data = AirbnbNightlyPriceImageDataset()

    # train_data, validation_data, test_data = random_split(data, [0.7, 0.15, 0.15])
    # batch_size = 64
    # data_loaders = {
    #     'train': torch.utils.data.DataLoader(
    #         train_data,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         pin_memory=torch.cuda.is_available(),
    #     ),
    #     'train_metrics': torch.utils.data.DataLoader(
    #         train_data,
    #         batch_size=len(train_data),
    #         shuffle=True,
    #         pin_memory=torch.cuda.is_available(),
    #     ),
    #     'validation': torch.utils.data.DataLoader(
    #         validation_data,
    #         batch_size=len(validation_data),
    #         shuffle=True,
    #         pin_memory=torch.cuda.is_available(),
    #     ),
    #     'test': torch.utils.data.DataLoader(
    #         test_data,
    #         batch_size=len(test_data),
    #         shuffle=True,
    #         pin_memory=torch.cuda.is_available(),
    #     )
    # }

    # model = NNRegression(config=nn_config)

    # model_metrics = evaluate_nn(model, data_loaders, nn_config, 40)

    # print()
    # print(model_metrics)
    # print()

    # save_nn(model, nn_config, model_metrics)
    
