from tabular_data import load_airbnb
from sklearn import linear_model, model_selection, metrics, preprocessing

import numpy as np
np.random.seed(20)

# Completes cross validation of a model on different parameters and returns the best model
def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, parameter_dictionary):

    pass

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

# Print R2 score on validation dataset
print(f'R^2 score: {model.score(X_val, y_val)}')
print()

# Print RMSE score on validation dataset
y_hat = model.predict(X_val)
print(f'RMSE score: {metrics.mean_squared_error(y_val, y_hat, squared=False)}')
print()

print(y_val)
print(y_hat)