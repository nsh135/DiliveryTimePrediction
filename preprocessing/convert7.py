from datetime import datetime, timedelta
from typing import List
import warnings, swifter, pickle, calendar
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from tensorflow.python.keras.backend import maximum
from misc import holidayDays
from preprocessing.zipcodes import distance_2zipcodes
import geopy.distance
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pandas as pd
import pandas.tseries.holiday

trainSetID = 7

global cacheZip, federalHolidays

def b2c_c2c_to_numeric(df):
    """
     convert the first column to numeric
     0 for b2c and 1 for c2c
    """
    def to_numeric(x):
        if (x =='B2C'): return 0
        elif (x =='C2C'): return 1
        else: 
            warnings.warn("Business Model not recognized")
            return None
    df['b2c_c2c'] = df['b2c_c2c'].swifter.apply(to_numeric)  # parallel apply
    return df


def distance_kernel(zipcodes):
    sourceZip, destZip = str(zipcodes).split('&')
    sourceZip = str(sourceZip)[:5]
    destZip = str(destZip)[:5]
    if sourceZip not in cacheZip or destZip not in cacheZip:
        return None
    return geopy.distance.geodesic(cacheZip[sourceZip], cacheZip[destZip]).miles

def calculate_distance(df:DataFrame, from_cache=True):
    """ calculate distance between source and destination by using zip code
        distance is then standardized with sklearn.preprocessing.StandarScaler
    """
    if from_cache:
        cache_df = pd.read_feather('preprocessing/distance_cache')
        df = pd.concat([df, cache_df[:df.shape[0]]], axis=1)
    else:
        zipcodes = df['item_zip'].astype(str) + '&' + df['buyer_zip'].astype(str)
        pool = mp.Pool(64)
        dist = list(tqdm(pool.imap(distance_kernel, zipcodes), total=len(zipcodes)))
        df['distance'] = dist
        # fill Nan with average
        df['distance'].fillna(value=df['distance'].mean(), inplace=True)
    df.drop('item_zip', axis=1)
    df.drop('buyer_zip', axis=1)
    return df

def cache_distance(df:DataFrame):
    """ calculate distance column and save to file """
    zipcodes = df['item_zip'].astype(str) + '&' + df['buyer_zip'].astype(str)
    pool = mp.Pool(64)
    dist = list(tqdm(pool.imap(distance_kernel, zipcodes), total=len(zipcodes)))
    df.drop('item_zip', axis=1)
    df.drop('buyer_zip', axis=1)
    df['distance'] = dist
    # fill Nan with average
    df['distance'].fillna(value=df['distance'].mean(), inplace=True)
    df[['distance']].to_feather('preprocessing/distance_cache')



def convert_weight(df:DataFrame):
    def convert_weight_kernel(series):
        weight, unit = series['weight'], series['weight_units']
        return float(weight) if int(unit)==1 else float(weight)*2.20462
    weights = df[['weight', 'weight_units']].swifter.apply(convert_weight_kernel, axis=1)
    df['weight'] = weights
    return df

def calculate_delivery_days(df:DataFrame):
    def delivery_days_kernel(series):
        accept_dt, deliver_dt = series['acceptance_scan_timestamp'], series['delivery_date']
        accept_dt = datetime.fromisoformat(accept_dt)
        deliver_dt = datetime.fromisoformat("{0} 17:00:00-07:00".format(deliver_dt))
        days = (deliver_dt.date()-accept_dt.date()).days
        return days if days>0 else None
    days = df[['acceptance_scan_timestamp', 'delivery_date']].swifter.apply(delivery_days_kernel, axis=1)
    return np.array(days)

def calculate_month(df:DataFrame):
    def month_kernel(acceptance_dt):
        acceptance_dt = datetime.fromisoformat(acceptance_dt)
        return acceptance_dt.month
    df['month'] = df['acceptance_scan_timestamp'].swifter.apply(month_kernel)
    return df

def calculate_day(df:DataFrame):
    def day_kernel(acceptance_dt):
        acceptance_dt = datetime.fromisoformat(acceptance_dt)
        return acceptance_dt.day
    df['day'] = df['acceptance_scan_timestamp'].swifter.apply(day_kernel)
    return df

def calculate_weekday(df:DataFrame):
    def weekday_kernel(acceptance_dt):
        acceptance_dt = datetime.fromisoformat(acceptance_dt)
        return calendar.day_abbr[acceptance_dt.weekday()]
    df['weekday'] = df['acceptance_scan_timestamp'].swifter.apply(weekday_kernel)
    return df

def calculate_hour(df:DataFrame):
    def hour_kernel(acceptance_dt):
        acceptance_dt = datetime.fromisoformat(acceptance_dt)
        return acceptance_dt.hour
    df['hour'] = df['acceptance_scan_timestamp'].swifter.apply(hour_kernel)
    return df


def calculate_holidays(df:DataFrame):
    def isHoliday_kernel(acceptance_dt):
        x = datetime.fromisoformat(acceptance_dt).date()
        return 1 if x in federalHolidays else 0
    def nearHoliday_kernel(acceptance_dt):
        x = datetime.fromisoformat(acceptance_dt).date()
        for d in range(1,8):
            if x+timedelta(days=d) in federalHolidays:
                return 1
        return 0
    df['is_holiday'] = df['acceptance_scan_timestamp'].swifter.apply(isHoliday_kernel)
    df['near_holiday'] = df['acceptance_scan_timestamp'].swifter.apply(nearHoliday_kernel)
    return df


def delivery_days_to_date(df, days) -> List:
    """ add y to df.acceptance_scan_timestamp 
    return Series of delivery dates (str)
    """
    assert df.shape[0] == len(days)
    # assert days.shape[1] == 1
    def kernel(series):
        acceptance_dt, days = series['acceptance_scan_timestamp'], series['days']
        max_carrier_est, min_carrier_est = series['carrier_max_estimate'], series['carrier_min_estimate']
        days = days if days>=0 else (0.4*min_carrier_est+0.6*max_carrier_est)
        days = min(days, max_carrier_est)
        acceptance_dt = datetime.fromisoformat(acceptance_dt)
        date = acceptance_dt + timedelta(days=days)
        return date.strftime('%Y-%m-%d')
    df['days'] = days
    results = df[['acceptance_scan_timestamp', 'days', 'carrier_min_estimate', 'carrier_max_estimate']].swifter.apply(kernel, axis=1)
    return pd.Series(list(results), name='delivery_date')
    

feature_names = ['b2c_c2c', 'shipment_method_id', 'shipping_fee', 'carrier_min_estimate', 'carrier_max_estimate', 'category_id', 'item_price', 'quantity', 'weight', 'distance', 'month', 'day', 'weekday', 'hour', 'is_holiday', 'near_holiday']
features_need_onehot = ['shipment_method_id','category_id', 'weekday']
onehot_categories = {
    'shipment_method_id': list(range(27)),
    'category_id': list(range(33)),
    # 'month': list(range(1,13)),
    # 'day': list(range(1,32)),
    'weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri','Sat', 'Sun'],
}

def onehot_encode(df:DataFrame):
    for onehot_feature in tqdm(features_need_onehot):
        col = onehot_categories[onehot_feature] + list(df[onehot_feature])
        temp_onehot_df = pd.get_dummies(col ,prefix=onehot_feature, drop_first=True)
        temp_onehot_df = temp_onehot_df.iloc[len(onehot_categories[onehot_feature]):]
        assert temp_onehot_df.shape[0] == df.shape[0]
        df = df.drop(onehot_feature, axis=1)
        temp_onehot_df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, temp_onehot_df], axis=1)
    return df

def clean_carrier_estimate(df:DataFrame):
    def kernel(x):
        return x if x>=0 else None
    df['carrier_min_estimate'] = df['carrier_min_estimate'].swifter.apply(kernel)
    df['carrier_max_estimate'] = df['carrier_max_estimate'].swifter.apply(kernel)
    return df

def process(df:DataFrame, no_target=False, record_number = False):
    """ return x, target """
    # getting cached data 
    global cacheZip, federalHolidays
    with open("data/preprocessing/zipcodes.pkl", "rb") as file:
        cacheZip = pickle.load(file)
    federalHolidays = holidayDays()

    # process data
    print("converting b2c ...")
    df = b2c_c2c_to_numeric(df)
    print("converting distance ...")
    if not no_target:
        df = calculate_distance(df, from_cache = True)
    else:
        df = calculate_distance(df, from_cache = False)
    print("converting weights ...")
    df = convert_weight(df)
    print("clearning carrier estimates ... ")
    df = clean_carrier_estimate(df)
    print("calculating acceptance months ...")
    df = calculate_month(df)
    print("calculating acceptance days ...")
    df = calculate_day(df)
    print("calculating acceptance weekday ...")
    df = calculate_weekday(df)
    print("calculating acceptance hour ...")
    df = calculate_hour(df)
    print("calculating holidays ... ")
    df = calculate_holidays(df)

    x = df[feature_names]
    print("one hot encoding ... ")
    x = onehot_encode(x)
    print("number of features in x: ", x.shape[1])
    if not no_target:
        print("calculating target ...")
        target = calculate_delivery_days(df)
    else:
        target = None

    if record_number: 
        record_number = df['record_number']
    else:
        record_number = None

    return x, target, record_number
