from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def c1_plots(adaboost_model, X_dev, X_test, y_dev, y_test, y_pred_test):
    # This function plots the MSE improvement over iterations and the actual vs predicted values for AdaBoost regression
    # In subtask 1, we used AdaBoost regression to predict the number of passengers boarding a bus at a given station.

    # Initialize lists to store MSE values for each iteration
    mse_dev_list = []
    mse_test_list = []
    # Calculate MSE for each iteration on dev and test sets
    for y_pred_dev_iter, y_pred_test_iter in zip(adaboost_model.staged_predict(X_dev),
                                                 adaboost_model.staged_predict(X_test)):
        mse_dev_list.append(mean_squared_error(y_dev, y_pred_dev_iter))
        mse_test_list.append(mean_squared_error(y_test, y_pred_test_iter))
    plt.figure(figsize=(10, 6))
    plt.plot(mse_dev_list, label='Dev MSE')
    plt.plot(mse_test_list, label='Test MSE')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Improvement Over Iterations')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('AdaBoost Regression: Actual vs Predicted')
    plt.show()

    c2_plots(y_test)


def c2_plots(y_test, history_dict, y_pred):
    # This function plots the MSE vs Epochs, Predicted vs Actual, and Residual Plot for the model in subtask 2
    # Plot MSE vs Epochs
    plt.figure(figsize=(12, 6))
    plt.plot(history_dict['mean_squared_error'], label='Training MSE')
    plt.plot(history_dict['val_mean_squared_error'], label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE vs. Epochs')
    plt.legend()
    plt.show()
    # Plot 2: Predicted vs Actual
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 100], [0, 100], color='red', linestyle='--')
    plt.xlabel('Actual Trip Duration')
    plt.ylabel('Predicted Trip Duration')
    plt.title('Predicted vs Actual Trip Duration')
    plt.show()
    # Plot 3: Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Trip Duration')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
