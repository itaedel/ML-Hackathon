import pandas as pd


def preprocess_test_data(data: pd.DataFrame):
    # Drop columns that are not useful or have been transformed
    data = data.drop(columns=['trip_id', 'part',
                              "trip_id_unique", "alternative", "station_name", "mekadem_nipuach_luz",
                              "latitude", "longitude", "cluster", "direction"])
    data = convert_time_columns(data)
    data = add_features(data)

    data = data.drop(columns=["arrival_time", "door_closing_time"])

    data = data.fillna(0)
    unique_station = data["trip_id_unique_station"]
    return data.drop(columns=["trip_id_unique_station"]), unique_station


def preprocess_data(data: pd.DataFrame):
    """
    Preprocess data for training

    Parameters
    ----------
    data: pd.DataFrame
        The input data

    Returns
    -------
    pd.DataFrame
        The preprocessed data

    """
    # drop duplicates
    data = data.drop_duplicates()

    # Drop columns that are not useful or have been transformed
    data = data.drop(columns=['trip_id', 'part',
                              "trip_id_unique", "alternative", "station_name", "mekadem_nipuach_luz",
                              "latitude", "longitude", "cluster", "direction"])

    # Non-negative values, times are in hh:mm format
    for c in ["arrival_time"]:
        data = data[(data[c] >= "00:00") & (data[c] <= "23:59")]

    # check station_index is non-negative
    data = data[data['station_index'] >= 0]

    # check passengers_up is non-negative
    data = data[data['passengers_up'] >= 0]

    # check passengers_continue is non-negative
    data = data[data['passengers_continue'] >= 0]

    data = convert_time_columns(data)
    data = add_features(data)

    data = data.drop(columns=["arrival_time", "door_closing_time"])
    data = handle_missing_values(data)
    unique_station = data["trip_id_unique_station"]
    return data.drop(columns=["trip_id_unique_station"]), unique_station


def add_features(data):
    # Define rush hour periods
    rush_hour_morning_start = pd.to_datetime('06:00:00', format='%H:%M:%S').time()
    rush_hour_morning_end = pd.to_datetime('09:00:00', format='%H:%M:%S').time()
    rush_hour_evening_start = pd.to_datetime('15:00:00', format='%H:%M:%S').time()
    rush_hour_evening_end = pd.to_datetime('19:00:00', format='%H:%M:%S').time()

    # Add rush_hour column
    data['rush_hour'] = data['arrival_time'].apply(
        lambda x: 1 if (
                (rush_hour_morning_start <= x.time() <= rush_hour_morning_end) or
                (rush_hour_evening_start <= x.time() <= rush_hour_evening_end)
        ) else 0
    )
    return data


def calculate_correlation(data):
    correlation = data['station_index'].corr(data['passengers_up'])
    print(f'Pearson correlation between station_index and passengers_up: {correlation}')


def handle_missing_values(data):
    return data.dropna()


def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-8')


def convert_time_to_minutes(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 60 + m


def convert_time_columns(data):
    data['door_closing_time'].fillna(data['arrival_time'], inplace=True)

    # Convert time columns to datetime format
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce')

    # Handle cases where the conversion to datetime fails (NaT values)
    data['arrival_time'].fillna(pd.Timestamp('00:00:00'), inplace=True)
    data['door_closing_time'].fillna(pd.Timestamp('00:00:00'), inplace=True)

    # Calculate the time the door was open
    data['door_open_time'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds() / 60

    # Handle negative or zero open times by setting them to a default value (e.g., 0)
    data['door_open_time'] = data['door_open_time'].apply(lambda x: x if x > 0 else 0)

    # remove rows with extreme values
    data = data[(data['door_open_time'] >= 0) & (data['door_open_time'] < 11)]
    # map true false on arrival_is_estimated
    data['arrival_is_estimated'] = data['arrival_is_estimated'].map({True: 1, False: 0}).astype('Int64')
    #data['arrival_is_estimated'] = data['arrival_is_estimated'].map({True: 1, False: 0})
    return data
