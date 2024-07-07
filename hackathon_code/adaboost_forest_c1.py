import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error


def create_adaboost_forest(data):
    # gets the pre-processed data
    # Split the data into features (X) and target variable (y)
    X = data.drop('passengers_up', axis=1)
    y = data['passengers_up']

    # Split the data into train, dev, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    # Create the Random Forest model with the best hyperparameters
    rf_model = RandomForestRegressor(n_estimators=20, max_depth=20, min_samples_split=5, min_samples_leaf=2,
                                     random_state=42)

    # Create the AdaBoost model with Random Forest as the base estimator and the best hyperparameters
    adaboost_model = AdaBoostRegressor(estimator=rf_model, n_estimators=20, learning_rate=0.1,
                                       random_state=42)

    # Fit the AdaBoost model on the training data
    adaboost_model.fit(X_train, y_train)

    # Make predictions on the dev set
    y_pred_dev = adaboost_model.predict(X_dev)

    # Calculate MSE on the dev set
    mse_dev = mean_squared_error(y_dev, y_pred_dev)
    print(f"Dev MSE: {mse_dev:.4f}")

    # Make predictions on the test set
    y_pred_test = adaboost_model.predict(X_test)

    # Calculate MSE on the test set
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Test MSE: {mse_test:.4f}")

    return adaboost_model

# MSE: 2.31
