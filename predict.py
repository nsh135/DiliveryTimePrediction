### produce results for quiz set
### by using model from train/saved_model

# select data processing function
import importlib
from misc import read_tsv
import tensorflow as tf
import pandas as pd
import csv, os
import argparse
import numpy
numpy.set_printoptions(threshold=200)


##only predict for a certain number of samples
nrows = None
import sys

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training a DNN model')
    parser.add_argument('--trainSetID', dest='trainSetID', type=int, help='trainSetID')
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # load model
    args = parser.parse_args()
    trainSetID = str(args.trainSetID)
    model_path = os.path.join('train/trainSet'+trainSetID,'saved_model' )
    model = tf.keras.models.load_model(model_path, compile=False)
    print("reading data file ... ")
    # load data and process
    convert = importlib.import_module('preprocessing.convert'+trainSetID)
    df = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_quiz.tsv", nrows=nrows)
    x, _, record_number = convert.process(df, no_target=True, record_number=True)
    print("Test data shape:", x.shape)
    # predict in days
    print("predicting ...")
    days = numpy.rint(model.predict(x, batch_size = 4096)[:,0])
    print("!!!!Predicted values: ", days)
    # convert days to date
    print("converting days to dates ... ")
    delivery_dates = convert.delivery_days_to_date(df, days)
    prediction_df = pd.concat([record_number,delivery_dates], axis=1)
    print(prediction_df.head())
    # write file according to requirements
    print("writing output ... ")
    prediction_df.to_csv("submissions/submission{}.tsv.gz".format(trainSetID), sep='\t', index=False, header=False, compression='gzip')
    

