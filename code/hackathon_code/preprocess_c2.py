import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_test_data(data):
    # Convert arrival_time to datetime for calculation purposes
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
    data['arrival_time'] = data['arrival_time'].fillna(pd.to_datetime('00:00:00', format='%H:%M:%S'))

    # Ensure that the data is sorted by trip_id_unique and station_index
    data = data.sort_values(by=['trip_id_unique', 'station_index'])

    # Function to fill missing values with the mean of forward and backward fill
    # def fill_missing_with_mean(group):
    #     forward_filled = group.fillna(method='ffill')
    #     backward_filled = group.fillna(method='bfill')
    #     group['latitude'] = (forward_filled['latitude'] + backward_filled['latitude']) / 2
    #     group['longitude'] = (forward_filled['longitude'] + backward_filled['longitude']) / 2
    #     return group
    #
    # # Group by 'trip_id_unique' and fill missing values with the mean of forward and backward fill
    # data = data.groupby('trip_id_unique').apply(fill_missing_with_mean).reset_index()
    # fill with mean
    data['latitude'] = data['latitude'].fillna(data['latitude'].mean())
    data['longitude'] = data['longitude'].fillna(data['longitude'].mean())

    # Debugging: Print the first few rows of data to check latitude and longitude values
    print("Data with latitude and longitude:")
    print(data[['trip_id_unique', 'station_index', 'latitude', 'longitude']].head())

    # Group by trip_id_unique to get the first and last arrival times
    trip_groups = data.groupby('trip_id_unique').agg(
        start_time=('arrival_time', 'first'),
        end_time=('arrival_time', 'last'),
        num_stops=('station_index', 'count')
    ).reset_index()

    # Calculate the trip duration in minutes
    trip_groups['trip_duration_in_minutes'] = (trip_groups['end_time'] - trip_groups[
        'start_time']).dt.total_seconds() / 60

    # Feature engineering
    trip_groups['start_hour'] = trip_groups['start_time'].dt.hour

    # Haversine function to calculate distance between two points in meters
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of the Earth in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    # Calculate average distance between stops (if latitude and longitude are available)
    if 'latitude' in data.columns and 'longitude' in data.columns:
        def calculate_distance(group):
            latitudes = group['latitude'].values
            longitudes = group['longitude'].values
            distances = [haversine(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1]) for i in
                         range(len(latitudes) - 1)]
            return np.mean(distances) if len(distances) > 0 else 0

        # Apply the distance calculation function to each group
        trip_distances = data.groupby('trip_id_unique').apply(calculate_distance).reset_index(
            name='avg_distance_between_stops')

        # Debugging: Print the first few rows of trip_distances to check calculated distances
        print("Calculated average distances between stops (in meters):")
        print(trip_distances.head())

        trip_groups = trip_groups.merge(trip_distances, on='trip_id_unique')

    # Debugging: Print the first few rows of trip_groups to check merged results
    print("Trip groups with average distances:")
    print(trip_groups.head())

    # Merge the features back to the main dataset
    data = data.merge(trip_groups, on='trip_id_unique')

    # Handle categorical variables
    categorical_columns = ['line_id', 'direction', 'cluster', 'alternative']
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col].astype(str))

    # Handle missing values
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')

    # Remove columns with no observed values
    valid_numerical_columns = [col for col in numerical_columns if data[col].notna().sum() > 0]
    data[valid_numerical_columns] = imputer.fit_transform(data[valid_numerical_columns])

    # Remove outliers (e.g., trips with duration > 3 standard deviations from the mean)
    mean_duration = data['trip_duration_in_minutes'].mean()
    std_duration = data['trip_duration_in_minutes'].std()
    data = data[(data['trip_duration_in_minutes'] > mean_duration - 3 * std_duration) &
                (data['trip_duration_in_minutes'] < mean_duration + 3 * std_duration)]

    # Select relevant features for prediction
    features = ['trip_id_unique', 'line_id', 'direction', 'alternative', 'cluster', 'num_stops', 'start_hour']

    # Add avg_distance_between_stops to the features list if it exists
    if 'avg_distance_between_stops' in trip_groups.columns:
        features.append('avg_distance_between_stops')

    # Ensure all selected features exist in the dataset
    features = [f for f in features if f in data.columns]

    # Prepare the final dataset
    X = data[features]
    #y = data['trip_duration_in_minutes']
    return X


# Load the provided CSV file
def preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-8')

    # Remove rows where arrival_time is missing
    data = data.dropna(subset=['arrival_time'])

    # Convert arrival_time to datetime for calculation purposes
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')

    # Ensure that the data is sorted by trip_id_unique and station_index
    data = data.sort_values(by=['trip_id_unique', 'station_index'])

    # Check if latitude and longitude columns are available and not null
    if 'latitude' in data.columns and 'longitude' in data.columns:
        data = data.dropna(subset=['latitude', 'longitude'])

    # Debugging: Print the first few rows of data to check latitude and longitude values
    print("Data with latitude and longitude:")
    print(data[['trip_id_unique', 'station_index', 'latitude', 'longitude']].head())

    # Group by trip_id_unique to get the first and last arrival times
    trip_groups = data.groupby('trip_id_unique').agg(
        start_time=('arrival_time', 'first'),
        end_time=('arrival_time', 'last'),
        num_stops=('station_index', 'count')
    ).reset_index()

    # Calculate the trip duration in minutes
    trip_groups['trip_duration_in_minutes'] = (trip_groups['end_time'] - trip_groups[
        'start_time']).dt.total_seconds() / 60

    # Feature engineering
    trip_groups['start_hour'] = trip_groups['start_time'].dt.hour

    # Haversine function to calculate distance between two points in meters
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of the Earth in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    # Calculate average distance between stops (if latitude and longitude are available)
    if 'latitude' in data.columns and 'longitude' in data.columns:
        def calculate_distance(group):
            latitudes = group['latitude'].values
            longitudes = group['longitude'].values
            distances = [haversine(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1]) for i in
                         range(len(latitudes) - 1)]
            return np.mean(distances) if len(distances) > 0 else 0

        # Apply the distance calculation function to each group
        trip_distances = data.groupby('trip_id_unique').apply(calculate_distance).reset_index(
            name='avg_distance_between_stops')

        # Debugging: Print the first few rows of trip_distances to check calculated distances
        print("Calculated average distances between stops (in meters):")
        print(trip_distances.head())

        trip_groups = trip_groups.merge(trip_distances, on='trip_id_unique')

    # Debugging: Print the first few rows of trip_groups to check merged results
    print("Trip groups with average distances:")
    print(trip_groups.head())

    # Merge the features back to the main dataset
    data = data.merge(trip_groups, on='trip_id_unique')

    # Handle categorical variables
    categorical_columns = ['line_id', 'direction', 'cluster', 'alternative']
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col].astype(str))

    # Handle missing values
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')

    # Remove columns with no observed values
    valid_numerical_columns = [col for col in numerical_columns if data[col].notna().sum() > 0]
    data[valid_numerical_columns] = imputer.fit_transform(data[valid_numerical_columns])

    # Remove outliers (e.g., trips with duration > 3 standard deviations from the mean)
    mean_duration = data['trip_duration_in_minutes'].mean()
    std_duration = data['trip_duration_in_minutes'].std()
    data = data[(data['trip_duration_in_minutes'] > mean_duration - 3 * std_duration) &
                (data['trip_duration_in_minutes'] < mean_duration + 3 * std_duration)]

    # Select relevant features for prediction
    features = ['trip_id_unique', 'line_id', 'direction', 'alternative', 'cluster', 'num_stops', 'start_hour']

    # Add avg_distance_between_stops to the features list if it exists
    if 'avg_distance_between_stops' in trip_groups.columns:
        features.append('avg_distance_between_stops')

    # Ensure all selected features exist in the dataset
    features = [f for f in features if f in data.columns]

    # Prepare the final dataset
    X = data[features]
    y = data['trip_duration_in_minutes']
    return X, y

    # Save the preprocessed data to CSV files
    # X.to_csv('preprocessed_features.csv', index=False, encoding='ISO-8859-8')
    # y.to_csv('preprocessed_target.csv', index=False, encoding='ISO-8859-8')

    # print("Preprocessing completed and saved to preprocessed_features.csv and preprocessed_target.csv")
