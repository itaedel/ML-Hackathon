import pandas as pd
import hackathon_code.preprocess_c1 as preprocess
import hackathon_code.adaboost_forest_c1 as adaboost_forest
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
    print(args)

    # 1. load the training set (args.training_set)
    data = preprocess.load_data(args.training_set)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    data, _ = preprocess.preprocess_data(data)

    # 3. train a model
    logging.info("training...")
    model = adaboost_forest.create_adaboost_forest(data)  # what is gets returned here????????

    # 4. load the test set (args.test_set)
    test_data = preprocess.load_data(args.test_set)

    # 5. preprocess the test set
    test_df, unique_station = preprocess.preprocess_test_data(test_data)
    logging.info("preprocessing test...")

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(test_df)

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    # concat the unique_station and predictions
    predictions_df = pd.DataFrame({"trip_id_unique_station": unique_station, "passengers_up": predictions})
    predictions_df.to_csv(args.out, index=False, encoding='ISO-8859-8')
