from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
import argparse, importlib, numpy, pickle
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from misc import loss, read_tsv
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import misc
from sklearn.metrics import make_scorer


nrows = 15000000



def csv2df(csv_file):
	"""
	csv_file: data input (.csv)
	return: pandas data frame 
	"""
	df = pd.read_csv(csv_file, sep=',', nrows= nrows, compression='gzip')
	return df

def tuneParams(X:DataFrame, y):
    """ print cv_results_ """
    param_grid = {
        'n_estimators': [80, 100, 120, 140],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'gamma': [6,7,8,9,10],
        'reg_alpha': [0.1, 0.15, 0.2]
    }
    def score(y_true, y_pred):
        return -misc.loss(y_true, y_pred)

    xgb = XGBRegressor(n_jobs=30, tree_method='gpu_hist')
    cv = GridSearchCV(xgb, param_grid, scoring=make_scorer(score), cv=2, verbose=4)
    cv.fit(X, y)
    file = open('cv_results', 'w')
    print("best parameter set: ", cv.best_params_, file=file)
    print("best loss: ", -cv.best_score_, file=file)
    print("all results:", file=file)
    print(cv.cv_results_, file=file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--trainSetID', dest='trainSetID', type=str, help='trainSetID number',default='11' )
    parser.add_argument('--predict', dest='predict', type=bool, help='make prediction on quiz', default=False)
    
    args = parser.parse_args()
    
    #get data file 
    print("Reading file..")
    df = csv2df("data/processed_data/trainSet{}.csv.gz".format(args.trainSetID))
    print("shape: ", df.shape)
    df = df.dropna()
    print("shape after drop NA: ", df.shape)

    # #prepare dataset   
    # # split to train and test 
    # print("splitting data ... ")
    # train, test = train_test_split(df, test_size=0.17, shuffle=True, random_state=10)
    # X_train = train.loc[:, train.columns != 'target_from_order_placement']
    # y_train = train[['target_from_order_placement']]
    # X_test = test.loc[:, test.columns != 'target_from_order_placement']
    # y_test = test[['target_from_order_placement']]
    X = df.loc[:, df.columns != 'target_from_order_placement']
    y = df[['target_from_order_placement']]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # train on full set
    xgb = XGBRegressor(n_jobs=30, tree_method='gpu_hist', gamma=6, learning_rate=0.15, n_estimators=140, reg_alpha=0.2)
    xgb.fit(X, y)

    # Cross-validation ... 
    # print("Cross-validation ....")
    # tuneParams(X, y)
    # # save model
    # with open("train/RF_trained.pkl", 'wb') as file:
    #     pickle.dump(pipeline, file)

    # y_test_pred = pipeline.predict(X_test)
    # loss_test = loss(y_test['target_from_order_placement'].tolist(), y_test_pred.tolist())
    # print('loss on test set: ', loss_test)

    print("predicting .... ")
    print("reading data file ... ")
    # load data and process
    convert = importlib.import_module('preprocessing.convert'+args.trainSetID)
    df = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_quiz.tsv", nrows=None)
    x, _, record_number = convert.process(df, no_target=True, record_number=True)
    print("Test data shape:", x.shape)
    # predict in days
    print("predicting ...")
    x = scaler.transform(x)
    y_pred = xgb.predict(x)
    days = numpy.rint(y_pred)
    print("!!!!Predicted values: ", days)
    # convert days to date
    print("converting days to dates ... ")
    delivery_dates = convert.delivery_days_to_date(df, days)
    prediction_df = pd.concat([record_number,delivery_dates], axis=1)
    print(prediction_df.head())
    # write file according to requirements
    print("writing output ... ")
    prediction_df.to_csv("submissions/xgboost.tsv.gz".format(args.trainSetID), sep='\t', index=False, header=False, compression='gzip')

    


