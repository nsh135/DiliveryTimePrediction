from datetime import datetime
import swifter, pickle
import pandas.tseries.holiday
import pandas as pd
from tqdm import tqdm
from misc import read_tsv
from preprocessing import convert8 as convert


####### target encoding ###################
## calculate mean(y) for each category of shipment_method_id, category_id, day, month, weekday, hour
## from train data set

def targetEncoding_findMean():
    train_df = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_train.tsv", nrows=None)
    print("calculating acceptance months ...")
    train_df = convert.calculate_month(train_df)
    print("calculating acceptance days ...")
    train_df = convert.calculate_day(train_df)
    print("calculating acceptance weekday ...")
    train_df = convert.calculate_weekday(train_df)
    print("calculating acceptance hour ...")
    train_df = convert.calculate_hour(train_df)
    print("calculating target ...")
    train_df['delivery_days'] = convert.calculate_delivery_days(train_df)

    means_shipment = train_df.groupby('shipment_method_id')['delivery_days'].mean()
    means_category = train_df.groupby('category_id')['delivery_days'].mean()
    means_day = train_df.groupby('day')['delivery_days'].mean()
    means_month = train_df.groupby('month')['delivery_days'].mean()
    means_weekday = train_df.groupby('weekday')['delivery_days'].mean()
    means_hour = train_df.groupby('hour')['delivery_days'].mean()
    return means_shipment, means_category, means_day, means_month, means_weekday, means_hour


##### find date range ##################### 
def find_date_range():
    def convertDate_kernel(date):
        return datetime.fromisoformat(date).date()

    train_df = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_train.tsv", nrows=None)
    train_dates = train_df['acceptance_scan_timestamp'].swifter.apply(convertDate_kernel)
    minTrain = min(train_dates)
    maxTrain = max(train_dates)
    
    quiz_df = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_quiz.tsv", nrows=None)
    quiz_dates = quiz_df['acceptance_scan_timestamp'].swifter.apply(convertDate_kernel)
    minQuiz = min(quiz_dates)
    maxQuiz = max(quiz_dates)

    print("trainSet date range: ", minTrain, maxTrain)
    print("quizSet date range: ", minQuiz, maxQuiz)



if __name__ == "__main__":
    find_date_range()
    # means_shipment, means_category, means_day, means_month, means_weekday, means_hour = targetEncoding_findMean()
    # print(means_category)
    # print(means_shipment)
    # print(means_day)
    # print(means_month)
    # print(means_weekday)
    # print(means_hour)
    # with open('preprocessing/category_means.pkl', 'wb') as file:
    #     pickle.dump((means_shipment, means_category, means_day, means_month, means_weekday, means_hour), file)
