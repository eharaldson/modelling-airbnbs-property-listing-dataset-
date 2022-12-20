from tabular_data import load_airbnb
from sklearn import linear_model, model_selection, metrics, preprocessing

import numpy as np
np.random.seed(20)

X, y = load_airbnb()    # Load data
X = np.array(X)
y = np.array(y).reshape(-1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)    # Split data into train, validation and test data
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

scaler = preprocessing.StandardScaler().fit(X_train)    # Set up a scaler to standardize data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

model = linear_model.SGDRegressor()   # Set up SGD (Stochastic gradient descent) linear regression model
model.fit(X_train, y_train)

print(model.score(X_val, y_val))
print()

predictions = model.predict(X_test)

print(predictions[:5])
print()
print(y_test[:5])