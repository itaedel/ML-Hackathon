import pandas as pd
from keras.src.callbacks import history
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor


def create_NLM_model(X, y):
    # # Load the preprocessed data
    # features_file_path = 'preprocessed_features.csv'
    # target_file_path = 'preprocessed_target.csv'
    #
    # features = pd.read_csv(features_file_path, encoding='ISO-8859-8')
    # target = pd.read_csv(target_file_path, encoding='ISO-8859-8')

    # Define features and target
    X = X.drop(columns=['trip_id_unique'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a function to create the neural network model
    def create_model(input_shape):
        model = Sequential()
        model.add(Dense(64, input_dim=input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        return model

    # Preprocess the training data to get the correct input shape for the model
    preprocessor = StandardScaler()
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # Define the input shape based on the preprocessed data
    input_shape = X_train_preprocessed.shape[1]

    # Wrap the Keras model with KerasRegressor
    keras_regressor = KerasRegressor(build_fn=create_model, input_shape=input_shape, epochs=50, batch_size=32,
                                     verbose=2)

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', keras_regressor)
    ])

    # Fit the model and capture the history
    # history = keras_regressor.fit(X_train_preprocessed, y_train, validation_split=0.2)

    # Extract the history dictionary
    # history_dict = history.history_

    # Directly fit the pipeline
    pipeline.fit(X_train, y_train)
    # plot_pipline(pipeline, X_test, y_test, X, history_dict)
    return pipeline


def plot_pipline(pipeline, X_test, y_test, X, history_dict):
    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model using MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

    # Predict the trip durations for the entire dataset
    trip_duration_predictions = pipeline.predict(X).flatten()

    # print the predictions object
    print(trip_duration_predictions)

    # # Prepare the output DataFrame
    # output_df = pd.DataFrame({
    #     'trip_id_unique': features['trip_id_unique'],
    #     'trip_duration_in_minutes': trip_duration_predictions
    # })
    #
    # # Save the output to a CSV file
    # output_file_path = 'trip_duration_predictions.csv'
    # output_df.to_csv(output_file_path, index=False, encoding='ISO-8859-8')

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

    print("Predictions saved to trip_duration_predictions.csv")
