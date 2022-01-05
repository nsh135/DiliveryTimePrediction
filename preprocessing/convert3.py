from datetime import datetime
import warnings, swifter, pickle
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from preprocessing.zipcodes import distance_2zipcodes
import geopy.distance
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

trainSetID = 3

global cacheZip
with open("data/preprocessing/zipcodes.pkl", "rb") as file:
    cacheZip = pickle.load(file)

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

def calculate_distance(df:DataFrame):
    """ calculate distance between source and destination by using zip code
        distance is then standardized with sklearn.preprocessing.StandarScaler
    """
    zipcodes = df['item_zip'].astype(str) + '&' + df['buyer_zip'].astype(str)
    pool = mp.Pool(64)
    dist = list(tqdm(pool.imap(distance_kernel, zipcodes), total=len(zipcodes)))
    df.drop('item_zip', axis=1)
    df.drop('buyer_zip', axis=1)
    df['distance'] = dist
    return df


def convert_weight(df:DataFrame):
    def convert_weight_kernel(series):
        weight, unit = series['weight'], series['weight_units']
        return float(weight) if int(unit)==1 else float(weight)*2.20462
    weights = df[['weight', 'weight_units']].swifter.apply(convert_weight_kernel, axis=1)
    df['weight'] = weights
    return df

def calculate_delivery_days(df:DataFrame):
    def delivery_days_kernel(series):
        payment_dt, deliver_dt = series['acceptance_scan_timestamp'], series['delivery_date']
        payment_dt = datetime.fromisoformat(payment_dt)
        deliver_dt = datetime.fromisoformat("{0} 17:00:00-07:00".format(deliver_dt))
        return abs((deliver_dt-payment_dt).days)
    days = df[['acceptance_scan_timestamp', 'delivery_date']].swifter.apply(delivery_days_kernel, axis=1)
    return np.array(days)


feature_names = ['b2c_c2c', 'declared_handling_days', 'shipment_method_id', 'shipping_fee', 'carrier_min_estimate', 'carrier_max_estimate', 'category_id', 'item_price', 'quantity', 'weight', 'distance']
def process(df:DataFrame):
    """ return x, target """
    # process data
    print("converting b2c ...")
    df = b2c_c2c_to_numeric(df)
    print("converting distance ...")
    df = calculate_distance(df)
    print("converting weights ...")
    df = convert_weight(df)
    x = df[feature_names]
    print("calculating target ...")
    target = calculate_delivery_days(df)
    return x, target
