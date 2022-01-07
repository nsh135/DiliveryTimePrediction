from datetime import datetime
import swifter, pickle
import pandas.tseries.holiday
import pandas as pd
from tqdm import tqdm
import numpy as np

#Read data from tsv
def read_tsv(tsv_file, nrows=None):
	"""
	tsv_file: raw data input (.tsv)
	return: pandas data frame 
	"""
	colnames = pd.read_csv(tsv_file, nrows=0).columns
	type_dict =  {
		'item_zip': 'str',
		'buyer_zip': 'str',
	}
	df = pd.read_csv(tsv_file, sep='\t', nrows=nrows,  dtype = type_dict)
	return df


def loss(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    assert len(y_true)==len(y_pred)
    N = len(y_true)
    Sum = 0
    for i in range(N):
        if y_pred[i]<y_true[i]: #late shipping
            Sum += 0.6*(y_true[i]-y_pred[i])
        elif y_pred[i]>y_true[i]:#early shipment
            Sum += 0.4*(y_pred[i]-y_true[i])
    return Sum/N


def holidayDays(start="2016-12-20", end="2020-04-30"):
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)
    cal = pandas.tseries.holiday.USFederalHolidayCalendar()
    holidays = cal.holidays(start=start, end=end)
    holidays = [d.date() for d in holidays]
    return set(holidays)


####### target encoding ###################
## calculate mean(y) for each category of shipment_method_id, category_id, day, month, weekday, hour
## from train data set
class targetEncoding:
    def __init__(self) -> None:
        global means_shipment, means_category, means_day, means_month, means_weekday, means_hour
        with open('preprocessing/category_means.pkl', 'rb') as file:
            means_shipment, means_category, means_day, means_month, means_weekday, means_hour = pickle.load(file)
    
    def encode_shipment(self, df:pd.DataFrame) -> pd.DataFrame:
        """ return pandas.Series """
        def kernel(x):
            return float(means_shipment[x])
        df['shipment_method_id'] = df['shipment_method_id'].swifter.apply(kernel)
        return df
    
    def encode_category(self, df:pd.DataFrame) -> pd.DataFrame:
        """ return pandas.Series """
        def kernel(x):
            return float(means_category[x])
        df['category_id'] = df['category_id'].swifter.apply(kernel)
        return df
    
    def encode_day(self, df:pd.DataFrame) -> pd.DataFrame:
        """ return pandas.Series """
        def kernel(x):
            return float(means_day[x])
        df['day'] = df['day'].swifter.apply(kernel)
        return df

    def encode_month(self, df:pd.DataFrame) -> pd.DataFrame:
        """ return pandas.Series """
        def kernel(x):
            return float(means_month[x])
        df['month'] = df['month'].swifter.apply(kernel)
        return df
    
    def encode_weekday(self, df:pd.DataFrame) -> pd.DataFrame:
        """ return pandas.Series """
        def kernel(x):
            return float(means_weekday[x])
        df['weekday'] = df['weekday'].swifter.apply(kernel)
        return df

    def encode_hour(self, df:pd.DataFrame) -> pd.DataFrame:
        """ return pandas.Series """
        def kernel(x):
            return float(means_hour[x])
        df['hour'] = df['hour'].swifter.apply(kernel)
        return df
