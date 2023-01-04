from tabular_data import load_airbnb
from sklearn import linear_model, model_selection, metrics, preprocessing, tree, ensemble, svm, neighbors
from itertools import product

import os
import json
import pickle
import numpy as np

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
def tune_regression_model_hyperparameters(estimator, X_train, y_train, X_test, y_test, hyperparameters):

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
def adaboost_CV(X_train, y_train, X_test, y_test, save_ = True):
    hyperparameters = {'estimator': [tree.DecisionTreeRegressor(), svm.NuSVR(), linear_model.SGDRegressor(), neighbors.KNeighborsRegressor()],
                   'n_estimators': [5, 20, 40, 80, 150, 250]}
    estimator = ensemble.AdaBoostRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, y_train, X_test, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/adaboost', model, model_hyperparameters, model_score_metrics)

# Function for running GridSearchCV for gradient boosting
def gradientboost_CV(X_train, y_train, X_test, y_test, save_ = True):
    hyperparameters = {'n_estimators': [5, 20, 40, 80, 150, 250],
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'learning_rate': [0.5, 1, 5, 10]}
    estimator = ensemble.GradientBoostingRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, y_train, X_test, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/gradient_boosting', model, model_hyperparameters, model_score_metrics)

# Function for running GridSearchCV for random forest
def random_forest_CV(X_train, y_train, X_test, y_test, save_ = True):
    hyperparameters = {'criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    'n_estimators': [5, 20, 50, 100, 200],
                    'max_depth': [1, 5, 25, 50, 100, 200],
                    'max_features' : ["sqrt", "log2", None], 
                    'min_samples_leaf': [1, 2, 3, 4, 5]}
    estimator = ensemble.RandomForestRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, y_train, X_test, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/random_forest', model, model_hyperparameters, model_score_metrics)

# Function for running GridSearchCV for decision tree regressor
def decision_tree_CV(X_train, y_train, X_test, y_test, save_ = True):
    hyperparameters = {'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    'max_depth': [1, 5, 25, 50, 100, 200]}
    estimator = tree.DecisionTreeRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimator, X_train, y_train, X_test, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/decision_tree', model, model_hyperparameters, model_score_metrics)

# Function for running GridSearchCV for sgd regressor
def sgd_regressor_CV(X_train, y_train, X_test, y_test, save_ = True):
    hyperparameters = {'penalty': ['l1', 'l2'],
                        'alpha': [10, 1, 0.1, 0.001, 0.0001, 0.00001, 0],
                        'max_iter': [1000, 10000, 100000, 1000000]}
    estimators = linear_model.SGDRegressor()
    model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(estimators, X_train, y_train, X_test, y_test, hyperparameters)

    if save_ == True:
        save_model('models/regression/linear_regression', model, model_hyperparameters, model_score_metrics)    

# Function for running GridSearchCV on all the regression models
def evaluate_all_models(X_train, y_train, X_test, y_test, save_ = True):

    # Run the regression models and save the best one after cross validation
    sgd_regressor_CV(X_train, y_train, X_test, y_test)
    decision_tree_CV(X_train, y_train, X_test, y_test)
    random_forest_CV(X_train, y_train, X_test, y_test)
    gradientboost_CV(X_train, y_train, X_test, y_test)
    adaboost_CV(X_train, y_train, X_test, y_test)

# Find the best model and return it
def find_best_model():

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


if __name__ == "__main__":

    # Load data for classification
    X, y = load_airbnb('Category')
    X = np.array(X)
    y = np.array(y).reshape(-1)

    print(X[:5,:])
    print()
    print(y[:5])
    # Split data into train, validation and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Set up a scaler to standardize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


