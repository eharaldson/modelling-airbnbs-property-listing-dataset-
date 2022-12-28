from tabular_data import load_airbnb
from sklearn import linear_model, model_selection, metrics, preprocessing, pipeline
from itertools import product

import os
import json
import pickle
import numpy as np
np.random.seed(20)

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
def tune_regression_model_hyperparameters(X_train, y_train, X_test, y_test):

    hyperparameters = {'penalty': ['l1', 'l2'],
                       'alpha': [10, 1, 0.1, 0.001, 0.0001, 0.00001, 0],
                       'max_iter': [1000, 10000, 100000, 1000000]}

    clf = model_selection.GridSearchCV(estimator=linear_model.SGDRegressor(),
                                       scoring='neg_root_mean_squared_error',
                                       param_grid=hyperparameters, 
                                       verbose=3,
                                       n_jobs=-1)

    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    model_hyperparamters = clf.best_params_
    validation_rmse_score = -clf.best_score_

    y_hat = clf.predict(X_test)
    test_rmse_score = -clf.score(X_test, y_test)
    test_r2_score = metrics.r2_score(y_test, y_hat)

    best_score_metrics = {'validation_RMSE': validation_rmse_score, 'test_RMSE': test_rmse_score, 'test_R2': test_r2_score}

    return best_model, model_hyperparamters, best_score_metrics

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

## Model 2: SGDRegressor =>
X, y = load_airbnb()
X = np.array(X)
y = np.array(y).reshape(-1)

# Split data into train, validation and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Set up a scaler to standardize data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model, model_hyperparameters, model_score_metrics = tune_regression_model_hyperparameters(X_train, y_train, X_test, y_test)
print(model)

save_model('models/regression/linear_regression', model, model_hyperparameters, model_score_metrics)

















## Model 1: LinearRegression =>
''' 
# Load data
X, y = load_airbnb()
X = np.array(X)
y = np.array(y).reshape(-1)

# Split data into train, validation and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

# Set up a scaler to standardize data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Set up linear regression model and fit to the data
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Print model coefficients
print(model.coef_)
print(model.intercept_)
print() 

# Print R2 score on validation dataset
print(f'R^2 score: {model.score(X_val, y_val)}')
print()

# Print RMSE score on validation dataset
y_hat = model.predict(X_val)
print(f'RMSE score: {metrics.mean_squared_error(y_val, y_hat, squared=False)}')
print()

# Print R2 score on test dataset
print(f'R^2 score on test: {model.score(X_test, y_test)}')
print()

# Print RMSE score on test dataset
y_hat = model.predict(X_test)
print(f'RMSE score on test: {metrics.mean_squared_error(y_test, y_hat, squared=False)}')
print()
'''