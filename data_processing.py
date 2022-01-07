## read .tsv file, convert to Pandas frame, preprocessing data and write output to csv under data/processed_data/
## right now, we only process the  first 5 records to be fast, will process the entire data at the end

import pandas as pd
import sys, pickle, tqdm
from preprocessing import convert12 as convert
from misc import read_tsv

pd.set_option('display.max_columns', None)

## number of rows to read, None for all
nrows = None

#write processed data frame to csv file
def write_csv(df, output_file_name):
	"""
	write a pandas frame to csv file 
	df: pandas data frame
	@ output 
	write output to csv file under data/processed_data 
	"""
	df.to_csv(output_file_name,index=False)
	

def read_train_data(id=1):
	with open("data/processed_data/trainSet{}".format(id), 'rb') as file:
		x, target = pickle.load(file)
	return x, target

from datetime import datetime

if __name__ == "__main__":
	print("convert ID ", convert.trainSetID)
	print("reading raw data ...")
	df = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_train.tsv", nrows=nrows)
	# print("Original file:\n",df.head())
	# process data here, e.g., add features , remove features, convert to numeric, etc. 
	# call this function on the quiz set to get data for submission
	x, target, _ = convert.process(df)
	df.dropna(inplace=True)
	print("data dimension: ", x.shape)

	# # write to pickle
	# print("writing to binary file ...")
	# with open("data/processed_data/trainSet{}".format(convert.trainSetID), 'wb') as file:
	# 	pickle.dump((x, target), file)

	# write to csv file
	print("writing to csv file ... ")
	processed_df = x
	processed_df['target_from_order_placement'] = target   	# add target
	processed_df.to_csv("data/processed_data/trainSet{}.csv.gz".format(convert.trainSetID), compression='gzip', index=False)
