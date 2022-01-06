from sklearn.model_selection import train_test_split
import argparse
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from misc import loss
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


nrows = None



def csv2df(csv_file):
	"""
	csv_file: data input (.csv)
	return: pandas data frame 
	"""
	df = pd.read_csv(csv_file, sep=',', nrows= nrows, compression='gzip')
	return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--train_file', dest='train_file', type=str, help='Path to trianing file (.csv)',default='data/processed_data/trainSet11.csv' )
    parser.add_argument('--predict', dest='predict', type=bool, help='make prediction on quiz', default=False)
    
    args = parser.parse_args()
    
    #get data file 
    print("Reading file..")
    df = csv2df(args.train_file)
    print("shape: ", df.shape)
    df = df.dropna()
    print("shape after drop NA: ", df.shape)

    #prepare dataset   
    # split to train and test 
    print("splitting data ... ")
    train, test = train_test_split(df, test_size=0.17, shuffle=True, random_state=10)
    X_train = train.loc[:, train.columns != 'target_from_order_placement']
    y_train = train[['target_from_order_placement']]
    X_test = test.loc[:, test.columns != 'target_from_order_placement']
    y_test = test[['target_from_order_placement']]

    # train
    print("training ....")
    pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(n_jobs=-1))])
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    loss_test = loss(y_test['target_from_order_placement'].tolist(), y_test_pred.tolist())
    print('loss on test set: ', loss_test)


