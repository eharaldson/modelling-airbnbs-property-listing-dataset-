from tabular_data import load_airbnb
from sklearn import linear_model, model_selection, metrics, preprocessing, tree, ensemble, svm, neighbors, datasets
from itertools import product

import os
import json
import pickle
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

# Function to save model
def save_model(folder, model, model_hyperparameters, model_score_metrics):

    cwd = os.getcwd()
    model_filename = os.path.join(cwd, folder+'/model.joblib')
    hyperparameter_filename = os.path.join(cwd, folder+'/hyperparameters.json')
    score_filename = os.path.join(cwd, folder+'/metrics.json')

    # Save model
    pickle.dump(model, open(model_filename, 'wb'))

    # Save hyperparameters
    with open(hyperparameter_filename, 'w') as outfile:
        json.dump(model_hyperparameters, outfile)

    # Save metrics
    with open(score_filename, 'w') as outfile:
        json.dump(model_score_metrics, outfile)

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

# Function for running GridSearchCV on all the regression models
def evaluate_all_regression_models(X_train, X_test, y_train, y_test, save_ = True):

    # Run the regression models and save the best one after cross validation
    sgd_regressor_CV(X_train, X_test, y_train, y_test)
    decision_tree_CV(X_train, X_test, y_train, y_test)
    random_forest_CV(X_train, X_test, y_train, y_test)
    gradientboost_CV(X_train, X_test, y_train, y_test)
    adaboost_CV(X_train, X_test, y_train, y_test)

# Find the best model and return it
def find_best_regression_model():

    cwd = os.getcwd()

    folder = os.path.join(cwd, 'models/regression')
    subfolders = ['adaboost', 'decision_tree', 'gradient_boosting', 'linear_regression', 'random_forest']

    best_score = 99999999999

    for subfolder in subfolders:

        sub_directory = os.path.join(folder, subfolder)
        metrics_path = os.path.join(sub_directory, 'metrics.json')

        with open(metrics_path, 'rb') as f:
            metrics = json.load(f)

        if metrics["validation_RMSE"] < best_score:
            best_score = metrics["validation_RMSE"]
            best_metrics = metrics

            model_path = os.path.join(sub_directory, 'model.joblib')
            with open(model_path, 'rb') as f:
                best_model = pickle.load(f)

            hyperparameter_path = os.path.join(sub_directory, 'hyperparameters.json')
            with open(hyperparameter_path, 'rb') as f:
                best_hyperparameters = json.load(f)

    return best_model, best_hyperparameters, best_metrics

''' Classification '''
# Function to turn labels into one hot encoded labels
def one_hot_encoder(labels, max_Labels = None):
        """ Takes in label encoded label data and one hot encoded labels. 

        Args:
            labels (array): the label data.
            max_labels (int): the number of classes.

        Returns:
            ohe labels: the one hot encoded labels.
        """
        if max_Labels == None:
            max_Labels = np.max(labels) + 1
        return np.eye(max_Labels)[labels]

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

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = generate_processed_data(regression=False)

    model, model_hyperparameters, model_score_metrics = gradientboost_classifier_CV(X_train, X_test, y_train, y_test, save_=True)

    print(model)
    print()
    print(model_hyperparameters)
    print()
    print(model_score_metrics)