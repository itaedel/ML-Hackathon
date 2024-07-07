import pandas as pd

import hackathon_code.preprocess_c2 as preprocess
import hackathon_code.train_and_predict_c2 as train_and_predict
from argparse import ArgumentParser
import logging

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set and preprocess (args.training_set)
    logging.info("loading train...")
    df_X, df_y = preprocess.preprocess_data(args.training_set)

    # 3. train a model
    logging.info("training...")
    model = train_and_predict.create_NLM_model(df_X, df_y)

    # 4. load the test set (args.test_set) and process it
    logging.info("loading test...")
    df_test = pd.read_csv(args.test_set, encoding='ISO-8859-8')
    df_test = preprocess.preprocess_test_data(df_test)
    logging.info("preprocessing test...")

    # 6. predict the test set using the trained model
    # logging.info("predicting...")
    uniqe_trip_id = df_test['trip_id_unique']
    y_pred = model.predict(df_test.drop(columns=['trip_id_unique']))
    res = pd.DataFrame({"trip_id_unique": uniqe_trip_id, 'trip_duration_in_minutes': y_pred})
    # drop duplicates
    res = res.drop_duplicates(subset='trip_id_unique')
    # 7. save the predictions to args.out
    res.to_csv(args.out, index=False, encoding='ISO-8859-8')
